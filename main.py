import asyncio
import io
import logging
import math
import zipfile
import time as time_module
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, time
import zoneinfo
from pathlib import Path
from typing import Optional, Union, Dict, List, Any

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import yfinance as yf
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from gymnasium import spaces
from pydantic import BaseModel
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Import your simulator and custom options environment
#from simulator import SPYStockSimulator  # Adjust import path if needed
#from options_env import SPYOptionsEnv     # Adjust import path if needed

app = FastAPI()

# IMPORTANT: Ensure CORS is enabled so React (localhost:3000 / 5173) can talk to FastAPI (localhost:8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Black-Scholes Helper ---
def black_scholes_call(S, K, T, r=0.05, sigma=0.18):
    """Calculates theoretical Call premium per share."""
    if T <= 0:
        return max(0.0, S - K)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    # Standard normal CDF approximation
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
    return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)

def black_scholes_put(S, K, T, r=0.05, sigma=0.18):
    """Calculates theoretical Put premium per share."""
    call = black_scholes_call(S, K, T, r, sigma)
    return call + K * math.exp(-r * T) - S

def compute_option_row(spot: float, strike: float, dte: int = 7, iv: float = 0.20):
    """
    Recalculates Option Premiums dynamically on every tick based on spot and DTE.
    """
    call_intrinsic = max(0.0, spot - strike)
    put_intrinsic = max(0.0, strike - spot)
    
    # Simple Black-Scholes extrinsic/time value approximation
    # Avoid zero division when dte is 0DTE
    t = max(dte, 0.5) / 365.0
    time_value = spot * iv * math.sqrt(t) * 0.4 * math.exp(-((spot - strike) / (spot * 0.05)) ** 2)
    
    call_mid = call_intrinsic + time_value
    put_mid = put_intrinsic + time_value
    
    spread = 0.10
    call_bid = max(0.01, round(call_mid - (spread / 2), 2))
    call_ask = max(0.02, round(call_mid + (spread / 2), 2))
    put_bid = max(0.01, round(put_mid - (spread / 2), 2))
    put_ask = max(0.02, round(put_mid + (spread / 2), 2))
    
    return {
        "strike": strike,
        "call_bid": call_bid,
        "call_ask": call_ask,
        "put_bid": put_bid,
        "put_ask": put_ask,
        "in_the_money_call": spot > strike,
        "in_the_money_put": spot < strike,
    }

def parse_sim_date(
    date_val: Optional[Union[str, date, datetime]]
) -> date:
    """Helper to reliably parse simulation dates into datetime.date objects."""
    # 1. Clean up string input (handle empty string, "null", "undefined", etc.)
    if isinstance(date_val, str):
        date_val = date_val.strip()
        if not date_val or date_val.lower() in ("none", "null", "undefined"):
            date_val = None

    # 2. Guard for None
    if date_val is None:
        # Changed warning -> info/debug so expected default route behavior doesn't trigger console warnings
        logging.info("parse_sim_date received None or empty value. Defaulting to system date.today().")
        return date.today()

    if isinstance(date_val, datetime):
        return date_val.date()
    if isinstance(date_val, date):
        return date_val

    try:
        return datetime.fromisoformat(str(date_val).replace("Z", "")).date()
    except ValueError:
        try:
            return datetime.strptime(str(date_val)[:10], "%Y-%m-%d").date()
        except ValueError:
            # Keep warning here because an unparseable string IS an actual error!
            logging.warning(f"parse_sim_date failed to parse '{date_val}'. Defaulting to date.today().")
            return date.today()


def calculate_dte_from_expiration(
    expiration_str: Optional[str], 
    current_sim_date: Optional[Union[str, date, datetime]] = None
) -> int:
    """
    Calculates Days To Expiration (DTE) relative to the simulator's current date.
    """
    if not expiration_str:
        return 7  # Fallback default
    
    try:
        exp_date = datetime.strptime(expiration_str[:10], "%Y-%m-%d").date()
        sim_today = parse_sim_date(current_sim_date)
        
        days_diff = (exp_date - sim_today).days
        return max(0, days_diff)  # Ensure non-negative DTE
    except ValueError:
        return 7

class SPYStockSimulator:
    def __init__(self, initial_price=100.0, annual_drift=0.08, base_volatility=0.16):
        """
        Emulates daily SPY price dynamics with mean-reverting drift,
        volatility clustering, and tail-risk jumps.
        """
        self.price = initial_price
        self.initial_price = initial_price
        
        # Base annual parameters (SPY historical averages: ~8-10% return, ~16% vol)
        self.mu_base = annual_drift
        self.sigma_base = base_volatility
        self.current_sigma = base_volatility
        
        # 1 trading day step
        self.dt = 1.0 / 252.0  
        
        # Volatility clustering memory (GARCH-like persistence)
        self.vol_persistence = 0.92
        
        # Jump Diffusion Parameters (Simulating sudden drops/rallies)
        self.jump_prob = 0.05       # 5% chance of a jump event on any given day
        self.jump_mean = -0.005     # Slight negative bias on jumps (market drops faster than it rises)
        self.jump_std = 0.025       # Magnitude of news shocks

    def step_day(self):
        """Advances the simulator by 1 trading day."""
        # 1. Update Volatility Clustering (stochastic volatility)
        vol_shock = np.random.normal(0, 1)
        # Volatility naturally mean-reverts to base_volatility while clustering
        self.current_sigma = max(
            0.08, 
            self.vol_persistence * self.current_sigma + 
            (1 - self.vol_persistence) * self.sigma_base + 
            0.02 * abs(vol_shock)
        )

        # 2. Mean-Reverting Drift Adjustment (prevents unrealistic infinite parabolic trends)
        deviation_from_base = (self.price - self.initial_price) / self.initial_price
        trend_pull = -0.05 * deviation_from_base  # Soft anchor to keep long-term growth realistic
        effective_mu = self.mu_base + trend_pull

        # 3. Standard GBM Diffusion
        epsilon = np.random.normal(0, 1)
        drift = (effective_mu - 0.5 * (self.current_sigma ** 2)) * self.dt
        diffusion = self.current_sigma * np.sqrt(self.dt) * epsilon

        # 4. Jump Shock Process (Poisson Jump Diffusion)
        jump = 0.0
        if np.random.rand() < self.jump_prob:
            jump = np.random.normal(self.jump_mean, self.jump_std)

        # 5. Calculate New Price
        daily_return = np.exp(drift + diffusion + jump)
        self.price = max(0.01, self.price * daily_return)

        return round(self.price, 2)

    def generate_bars(self, num_days=252):
        """Convenience method to generate a full array of daily closing prices."""
        prices = [self.price]
        for _ in range(num_days):
            prices.append(self.step_day())
        return np.array(prices)

class QQQStockSimulator:
    def __init__(self, initial_price=100.0, annual_drift=0.08, base_volatility=0.16):
        """
        Emulates daily SPY price dynamics with mean-reverting drift,
        volatility clustering, and tail-risk jumps.
        """
        self.price = initial_price
        self.initial_price = initial_price
        
        # Base annual parameters (SPY historical averages: ~8-10% return, ~16% vol)
        self.mu_base = annual_drift
        self.sigma_base = base_volatility
        self.current_sigma = base_volatility
        
        # 1 trading day step
        self.dt = 1.0 / 252.0  
        
        # Volatility clustering memory (GARCH-like persistence)
        self.vol_persistence = 0.92
        
        # Jump Diffusion Parameters (Simulating sudden drops/rallies)
        self.jump_prob = 0.05       # 5% chance of a jump event on any given day
        self.jump_mean = -0.005     # Slight negative bias on jumps (market drops faster than it rises)
        self.jump_std = 0.025       # Magnitude of news shocks

    def step_day(self):
        """Advances the simulator by 1 trading day."""
        # 1. Update Volatility Clustering (stochastic volatility)
        vol_shock = np.random.normal(0, 1)
        # Volatility naturally mean-reverts to base_volatility while clustering
        self.current_sigma = max(
            0.08, 
            self.vol_persistence * self.current_sigma + 
            (1 - self.vol_persistence) * self.sigma_base + 
            0.02 * abs(vol_shock)
        )

        # 2. Mean-Reverting Drift Adjustment (prevents unrealistic infinite parabolic trends)
        deviation_from_base = (self.price - self.initial_price) / self.initial_price
        trend_pull = -0.05 * deviation_from_base  # Soft anchor to keep long-term growth realistic
        effective_mu = self.mu_base + trend_pull

        # 3. Standard GBM Diffusion
        epsilon = np.random.normal(0, 1)
        drift = (effective_mu - 0.5 * (self.current_sigma ** 2)) * self.dt
        diffusion = self.current_sigma * np.sqrt(self.dt) * epsilon

        # 4. Jump Shock Process (Poisson Jump Diffusion)
        jump = 0.0
        if np.random.rand() < self.jump_prob:
            jump = np.random.normal(self.jump_mean, self.jump_std)

        # 5. Calculate New Price
        daily_return = np.exp(drift + diffusion + jump)
        self.price = max(0.01, self.price * daily_return)

        return round(self.price, 2)

    def generate_bars(self, num_days=252):
        """Convenience method to generate a full array of daily closing prices."""
        prices = [self.price]
        for _ in range(num_days):
            prices.append(self.step_day())
        return np.array(prices)

def calculate_trend_signal(price_history, window=20):
    """
    Calculates normalized trend signal: (Price - SMA_20) / SMA_20
    - Positive values (> 0.0) indicate an uptrend (bullish momentum).
    - Negative values (< 0.0) indicate a downtrend (bearish momentum).
    """
    if len(price_history) < window:
        # Fallback for initial steps before full 20-day history is built
        window = len(price_history)
        
    sma_20 = np.mean(price_history[-window:])
    current_price = price_history[-1]
    
    trend_signal = (current_price - sma_20) / sma_20
    
    # Clip extreme values to prevent exploding gradients in RL observation space
    return float(np.clip(trend_signal, -0.10, 0.10))

# Daily Stock Simulator (SPY)
#spy_sim = SPYStockSimulator(initial_price=450.0, annual_drift=0.09, base_volatility=0.18)
#spy_recent_prices = [spy_sim.price]

# 5-Minute Intraday Stock Simulator (e.g., QQQ)
#qqq_sim = QQQStockSimulator(initial_price=380.0, base_volatility=0.22)
#qqq_recent_prices = [qqq_sim.price]

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

# -------------------------------------------------------------------
# 1. HELPER FUNCTIONS & TIME MANAGEMENT
# -------------------------------------------------------------------

def parse_sim_date(
    date_val: Optional[Union[str, date, datetime]]
) -> date:
    """Helper to reliably parse simulation dates into datetime.date objects."""
    # 1. Clean up string input (handle empty string, "null", "undefined", etc.)
    if isinstance(date_val, str):
        date_val = date_val.strip()
        if not date_val or date_val.lower() in ("none", "null", "undefined"):
            date_val = None

    # 2. Guard for None
    if date_val is None:
        # Changed warning -> info/debug so expected default route behavior doesn't trigger console warnings
        logging.info("parse_sim_date received None or empty value. Defaulting to system date.today().")
        return date.today()

    if isinstance(date_val, datetime):
        return date_val.date()
    if isinstance(date_val, date):
        return date_val

    try:
        return datetime.fromisoformat(str(date_val).replace("Z", "")).date()
    except ValueError:
        try:
            return datetime.strptime(str(date_val)[:10], "%Y-%m-%d").date()
        except ValueError:
            # Keep warning here because an unparseable string IS an actual error!
            logging.warning(f"parse_sim_date failed to parse '{date_val}'. Defaulting to date.today().")
            return date.today()


def step_5m(current_dt: datetime) -> datetime:
    """Advances time by 5 minutes, skipping non-trading hours and weekends."""
    next_dt = current_dt + timedelta(minutes=5)
    
    # After 16:00 market close -> Jump to 09:30 next day
    if next_dt.time() > time(16, 0):
        next_dt = datetime.combine(next_dt.date() + timedelta(days=1), time(9, 30))

    # Skip weekends
    while next_dt.weekday() in (5, 6):
        next_dt = datetime.combine(next_dt.date() + timedelta(days=1), time(9, 30))

    return next_dt


def step_1d(current_dt: datetime) -> datetime:
    """Advances time by 1 day, skipping weekends."""
    next_dt = current_dt + timedelta(days=1)
    while next_dt.weekday() in (5, 6):
        next_dt += timedelta(days=1)
    return next_dt


def load_ppo_model_safe(zip_filename: str, vec_env) -> PPO:
    """Safely reads policy.pth into an in-memory buffer bypassing Windows zip errors."""
    zip_path = MODELS_DIR / zip_filename
    model = PPO("MlpPolicy", vec_env, device="cpu")

    with zipfile.ZipFile(zip_path, "r") as z:
        policy_bytes = z.read("policy.pth")

    buffer = io.BytesIO(policy_bytes)
    state_dict = torch.load(buffer, map_location="cpu", weights_only=False)
    model.policy.load_state_dict(state_dict)
    return model


# -------------------------------------------------------------------
# 2. MODULAR STOCK REGISTRY ARCHITECTURE
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# MODULAR STOCK REGISTRY ARCHITECTURE (UPDATED)
# -------------------------------------------------------------------

@dataclass
class StockContext:
    symbol: str
    interval: str  # "1d" or "5m"
    simulator: Any
    bull_env: Any
    bear_env: Any
    bull_npc: PPO
    bull_norm: VecNormalize
    bear_npc: PPO
    bear_norm: VecNormalize
    current_dt: datetime
    initial_dt: datetime  # STORE INITIAL DATE TO RESTORE ON RESET
    recent_prices: List[float] = field(default_factory=list)

    def reset_state(self):
        """Resets price buffers and restores initial datetime for websocket re-connection."""
        init_price = self.simulator.price
        self.recent_prices = [init_price] * 300
        
        # Reset Environments
        if hasattr(self.bull_env, "spy_prices"):
            self.bull_env.spy_prices = self.recent_prices.copy()
            self.bear_env.spy_prices = self.recent_prices.copy()
        else:
            self.bull_env.stock_prices = self.recent_prices.copy()
            self.bear_env.stock_prices = self.recent_prices.copy()

        self.bull_env.reset()
        self.bear_env.reset()

        # FIX: Restore to saved initial_dt instead of regenerating datetime.now()
        self.current_dt = self.initial_dt

    def step(self) -> float:
        """Ticks simulator and advances timestamp regardless of active UI tab."""
        if self.interval == "5m":
            self.current_dt = step_5m(self.current_dt)
            price = float(self.simulator.step_5m() if hasattr(self.simulator, 'step_5m') else self.simulator.step_day())
        else:
            self.current_dt = step_1d(self.current_dt)
            price = float(self.simulator.step_day())

        self.recent_prices.append(price)
        if len(self.recent_prices) > 300:
            self.recent_prices.pop(0)

        # Sync updated prices into environments
        if hasattr(self.bull_env, "spy_prices"):
            self.bull_env.spy_prices = self.recent_prices.copy()
            self.bear_env.spy_prices = self.recent_prices.copy()
        else:
            self.bull_env.stock_prices = self.recent_prices.copy()
            self.bear_env.stock_prices = self.recent_prices.copy()

        return price

    def get_formatted_time(self) -> str:
        # 1. Ensure datetime has timezone info (assume UTC if naive)
        if self.current_dt.tzinfo is None:
            utc_dt = self.current_dt.replace(tzinfo=zoneinfo.ZoneInfo("UTC"))
        else:
            utc_dt = self.current_dt

        # 2. Convert to US Eastern Time (handles EST/EDT daylight saving shifts automatically)
        eastern_dt = utc_dt.astimezone(zoneinfo.ZoneInfo("America/New_York"))

        # 3. Format string based on chart interval requirement
        if self.interval == "5m":
            return eastern_dt.strftime("%Y-%m-%d %H:%M:%S")
        
        return eastern_dt.strftime("%Y-%m-%d")


def build_stock_context(
    symbol: str,
    interval: str,
    simulator: Any,
    bull_env_cls: type,
    bear_env_cls: type,
    bull_model_file: str,
    bull_norm_file: str,
    bear_model_file: str,
    bear_norm_file: str
) -> StockContext:
    """Factory builder for registering any new stock into the runtime context."""
    init_price = simulator.price
    prices_buffer = [init_price] * 300

    bull_env = bull_env_cls(spy_prices=prices_buffer) if "spy_prices" in bull_env_cls.__init__.__code__.co_varnames else bull_env_cls(stock_prices=prices_buffer)
    bear_env = bear_env_cls(spy_prices=prices_buffer) if "spy_prices" in bear_env_cls.__init__.__code__.co_varnames else bear_env_cls(stock_prices=prices_buffer)

    bull_vec_env = DummyVecEnv([lambda: bull_env])
    bear_vec_env = DummyVecEnv([lambda: bear_env])

    bull_npc = load_ppo_model_safe(bull_model_file, bull_vec_env)
    bull_norm = VecNormalize.load(str(MODELS_DIR / bull_norm_file), bull_vec_env)
    bull_norm.training = False

    bear_npc = load_ppo_model_safe(bear_model_file, bear_vec_env)
    bear_norm = VecNormalize.load(str(MODELS_DIR / bear_norm_file), bear_vec_env)
    bear_norm.training = False

    # Define base start date (e.g. 300 days in past for daily data to allow history)
    start_base_date = datetime.now().date() - timedelta(days=300 if interval == "1d" else 0)
    init_dt = datetime.combine(start_base_date, time(9, 30) if interval == "5m" else time(0, 0))

    return StockContext(
        symbol=symbol,
        interval=interval,
        simulator=simulator,
        bull_env=bull_env,
        bear_env=bear_env,
        bull_npc=bull_npc,
        bull_norm=bull_norm,
        bear_npc=bear_npc,
        bear_norm=bear_norm,
        current_dt=init_dt,
        initial_dt=init_dt,  # PASS STORED INITIAL DT
        recent_prices=prices_buffer
    )

# --- GLOBAL STOCKS REGISTRY ---
# TO ADD A NEW STOCK: Simply instantiate simulator + add 1 entry to STOCKS registry.
STOCKS: Dict[str, StockContext] = {
    "SPY": build_stock_context(
        symbol="SPY",
        interval="1d",
        simulator=SPYStockSimulator(initial_price=450.0, annual_drift=0.09, base_volatility=0.18),
        bull_env_cls=SPYOptionsEnvCalls,
        bear_env_cls=SPYOptionsEnvPuts,
        bull_model_file="ppo_spy_options_bull_specialist.zip",
        bull_norm_file="vec_normalize_bull_specialist.pkl",
        bear_model_file="ppo_spy_options_bear_specialist.zip",
        bear_norm_file="vec_normalize_bear_specialist.pkl",
    ),
    "QQQ": build_stock_context(
        symbol="QQQ",
        interval="5m",
        simulator=QQQStockSimulator(initial_price=380.0, base_volatility=0.22),
        bull_env_cls=QQQOptionsEnvCalls,
        bear_env_cls=QQQOptionsEnvPuts,
        bull_model_file="ppo_spy_options_bull_specialist.zip",
        bull_norm_file="vec_normalize_bull_specialist.pkl",
        bear_model_file="ppo_spy_options_bear_specialist.zip",
        bear_norm_file="vec_normalize_bear_specialist.pkl",
    )
}


# -------------------------------------------------------------------
# 3. AGENT STEPPING HELPERS
# -------------------------------------------------------------------

def parse_npc_action(agent_name: str, action_vec: Any, current_price: float, env: Any, prev_contracts: int = 0) -> str:
    action_type = int(action_vec[0]) if isinstance(action_vec, (list, tuple, np.ndarray)) else int(action_vec)
    price_str = f"${current_price:.2f}"

    last_event = getattr(env, "last_event", None)
    if last_event == "STOP_LOSS":
        return f"{agent_name}: LIQUIDATED (Stop-Loss Hit) @ {price_str}"
    elif last_event == "EXPIRED":
        return f"{agent_name}: POSITION EXPIRED @ {price_str}"

    if action_type in [1, 2] and env.contracts > 0 and prev_contracts == 0:
        opt_type = "CALL" if getattr(env, "pos_type", 1) == 1 or action_type == 1 else "PUT"
        return f"{agent_name}: BOUGHT {opt_type} @ {price_str}"

    if prev_contracts > 0 and env.contracts == 0:
        return f"{agent_name}: CLOSED POSITION @ {price_str}"

    if env.contracts > 0:
        pos_label = "Call" if getattr(env, "pos_type", 1) == 1 else "Put"
        return f"{agent_name}: HELD {env.contracts} {pos_label} contract(s) @ {price_str}"

    return f"{agent_name}: HELD CASH @ {price_str}"


def step_agent(npc: PPO, norm_env: VecNormalize, env: Any, current_price: float, agent_name: str) -> dict:
    prev_contracts = env.contracts
    raw_obs = env._get_normalized_observation()
    norm_obs = norm_env.normalize_obs(np.array([raw_obs]))

    action, _ = npc.predict(norm_obs, deterministic=True)
    act_vec = action[0] if action.ndim > 1 else action

    env.step(act_vec)

    action_log = parse_npc_action(agent_name, act_vec, current_price, env, prev_contracts=prev_contracts)
    portfolio_val = env._get_portfolio_value(current_price)

    return {
        "action": action_log,
        "portfolio_value": round(float(portfolio_val), 2),
        "cash": round(float(env.cash), 2),
        "contracts": int(env.contracts),
    }


# -------------------------------------------------------------------
# 4. MODULAR REST API ENDPOINTS
# -------------------------------------------------------------------

@app.get("/api/options-expirations")
def get_options_expirations(
    symbol: str = Query("SPY", description="Stock Symbol"),
    simulated_date: Optional[str] = Query(None, alias="simulated_date")
):
    sim_today = parse_sim_date(simulated_date) if simulated_date else STOCKS.get(symbol, STOCKS["SPY"]).current_dt
    sample_offsets = [0, 1, 7, 14, 30, 45, 60]
    expirations = [
        (sim_today + timedelta(days=days)).strftime("%Y-%m-%d")
        for days in sample_offsets
    ]
    return expirations


@app.get("/api/options-chain")
def get_options_chain(
    symbol: str = Query("SPY", description="Stock Symbol"),
    current_price: float = Query(..., description="Spot price from WebSocket"),
    expiration: Optional[str] = Query(None, description="ISO Expiration date (YYYY-MM-DD)")
):
    dte = calculate_dte_from_expiration(expiration)

    # Dynamic strike spacing based on stock price magnitude
    strike_step = 2.5 if current_price < 500 else 5.0
    center_strike = round(current_price / strike_step) * strike_step
    strikes = [round(center_strike + (i * strike_step), 2) for i in range(-5, 6)]

    chain = [compute_option_row(current_price, strike, dte=dte) for strike in strikes]
    return {"symbol": symbol, "chain": chain}


# -------------------------------------------------------------------
# 5. WEBSOCKET LOOP (PARALLEL TICKING FOR ALL STOCKS)
# -------------------------------------------------------------------

@app.websocket("/ws/stocks")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Frontend client connected!")

    # Reset state for all registered stock contexts
    for ctx in STOCKS.values():
        ctx.reset_state()

    try:
        while True:
            dates_payload = {}
            data_payload = {}
            npcs_payload = {}
            active_logs = []

            # Tick ALL registered stocks in parallel every iteration
            for symbol, ctx in STOCKS.items():
                price = ctx.step()

                # Step Agents for this stock
                bull_data = step_agent(ctx.bull_npc, ctx.bull_norm, ctx.bull_env, price, f"{symbol}_Bull_NPC")
                bear_data = step_agent(ctx.bear_npc, ctx.bear_norm, ctx.bear_env, price, f"{symbol}_Bear_NPC")

                # Record Payloads
                dates_payload[symbol] = ctx.get_formatted_time()
                data_payload[symbol] = round(price, 2)
                npcs_payload[symbol] = {
                    "bull": bull_data,
                    "bear": bear_data
                }

                # Collect Active Trade Logs across all stocks
                # Include ALL actions, or add "HELD" to the keyword filter
                for data in [bull_data, bear_data]:
                    action_str = data.get("action", "")
                    # Add HELD if you want continuous updates every 3 seconds
                    if any(kw in action_str for kw in ["BOUGHT", "CLOSED", "LIQUIDATED", "EXPIRED", "HELD"]):
                        active_logs.append({
                            "text": action_str,
                            "symbol": symbol
                        })

            # Consolidated Packet for ALL symbols
            packet = {
                "timestamp": int(time_module.time()),
                "dates": dates_payload,
                "data": data_payload,
                "npcs": npcs_payload,
                "logs": active_logs
            }

            await websocket.send_json(packet)
            await asyncio.sleep(3)

    except WebSocketDisconnect:
        print("Frontend client disconnected cleanly.")
    except Exception as e:
        print(f"❌ Error in WebSocket loop: {e}")
        import traceback
        traceback.print_exc()
        await websocket.close()