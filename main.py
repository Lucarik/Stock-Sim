import asyncio
import io
import logging
import math
import zipfile
import time as time_module
from datetime import date, datetime, timedelta, time
from pathlib import Path
from typing import Optional, Union

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


@app.get("/api/options-expirations")
def get_options_expirations(
    simulated_date: Optional[str] = Query(None, alias="simulated_date")
):
    sim_today = parse_sim_date(simulated_date)

    sample_offsets = [0, 1, 7, 14, 30, 45, 60]
    expirations = [
        (sim_today + timedelta(days=days)).strftime("%Y-%m-%d")
        for days in sample_offsets
    ]
    return expirations

# 2. Updated options chain route accepting `expiration`
@app.get("/api/options-chain")
def get_options_chain(
    current_price: float = Query(..., description="Spot price from WebSocket"),
    expiration: Optional[str] = Query(None, description="ISO Expiration date (YYYY-MM-DD)")
):
    # Compute actual DTE from the passed expiration string
    dte = calculate_dte_from_expiration(expiration)

    print(f"[API TICK RECEIVED] Spot: {current_price} | Expiration: {expiration} (Calculated DTE: {dte})")
    
    # Generate strikes dynamically around the current spot price
    center_strike = round(current_price / 2.5) * 2.5
    strikes = [round(center_strike + (i * 2.5), 2) for i in range(-5, 6)]
    
    # Compute option row passing the calculated DTE
    chain = [compute_option_row(current_price, strike, dte=dte) for strike in strikes]
    
    return {"chain": chain}

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

# Daily Stock Simulator (SPY)
spy_sim = SPYStockSimulator(initial_price=450.0, annual_drift=0.09, base_volatility=0.18)
spy_recent_prices = [spy_sim.price]

# 5-Minute Intraday Stock Simulator (e.g., QQQ)
qqq_sim = QQQStockSimulator(initial_price=380.0, base_volatility=0.22)
qqq_recent_prices = [qqq_sim.price]


# Helper: Advance 5-minute timestamps skipping overnight & weekends
def step_5m(current_dt: datetime) -> datetime:
    next_dt = current_dt + timedelta(minutes=5)

    # Market closes after 16:00 -> Jump to next day's open at 09:30
    if next_dt.time() > time(16, 0):  # Changed >= to > so 16:00 is preserved
        next_dt = datetime.combine(next_dt.date() + timedelta(days=1), time(9, 30))

    # If new day lands on Weekend (Saturday/Sunday) -> Jump to Monday 09:30
    while next_dt.weekday() in (5, 6):
        next_dt = datetime.combine(next_dt.date() + timedelta(days=1), time(9, 30))

    return next_dt


# -------------------------------------------------------------------
# 2. INSTANTIATE ENVIRONMENTS & LOAD MODELS
# -------------------------------------------------------------------

# --- SPY Environments (Daily) ---
spy_bull_env = SPYOptionsEnvCalls(spy_prices=[spy_sim.price] * 300)
spy_bear_env = SPYOptionsEnvPuts(spy_prices=[spy_sim.price] * 300)
spy_bull_env.reset()
spy_bear_env.reset()

spy_bull_vec_env = DummyVecEnv([lambda: spy_bull_env])
spy_bear_vec_env = DummyVecEnv([lambda: spy_bear_env])

# --- QQQ Environments (5-Minute Intraday) ---
qqq_bull_env = QQQOptionsEnvCalls(spy_prices=[qqq_sim.price] * 300)
qqq_bear_env = QQQOptionsEnvPuts(spy_prices=[qqq_sim.price] * 300)
qqq_bull_env.reset()
qqq_bear_env.reset()
qqq_bull_vec_env = DummyVecEnv([lambda: qqq_bull_env])
qqq_bear_vec_env = DummyVecEnv([lambda: qqq_bear_env])

# Resolve base directory relative to main.py
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

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


# Load SPY Agents
spy_bull_npc = load_ppo_model_safe("ppo_spy_options_bull_specialist.zip", spy_bull_vec_env)
spy_bull_norm = VecNormalize.load(str(MODELS_DIR / "vec_normalize_bull_specialist.pkl"), spy_bull_vec_env)
spy_bull_norm.training = False

spy_bear_npc = load_ppo_model_safe("ppo_spy_options_bear_specialist.zip", spy_bear_vec_env)
spy_bear_norm = VecNormalize.load(str(MODELS_DIR / "vec_normalize_bear_specialist.pkl"), spy_bear_vec_env)
spy_bear_norm.training = False

# Load QQQ 5-Minute Agents
qqq_bull_npc = load_ppo_model_safe("ppo_spy_options_bull_specialist.zip", qqq_bull_vec_env)
qqq_bull_norm = VecNormalize.load(str(MODELS_DIR / "vec_normalize_bull_specialist.pkl"), qqq_bull_vec_env)
qqq_bull_norm.training = False

qqq_bear_npc = load_ppo_model_safe("ppo_spy_options_bear_specialist.zip", qqq_bear_vec_env)
qqq_bear_norm = VecNormalize.load(str(MODELS_DIR / "vec_normalize_bear_specialist.pkl"), qqq_bear_vec_env)
qqq_bear_norm.training = False


# -------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# -------------------------------------------------------------------

def parse_npc_action(agent_name, action_vec, current_price, env, prev_contracts=0):
    action_type = int(action_vec[0]) if isinstance(action_vec, (list, tuple, np.ndarray)) else int(action_vec)
    price_str = f"${current_price:.2f}"

    last_event = getattr(env, "last_event", None)
    if last_event == "STOP_LOSS":
        return f"{agent_name}: LIQUIDATED (Stop-Loss Hit) @ {price_str}"
    elif last_event == "EXPIRED":
        return f"{agent_name}: POSITION EXPIRED @ {price_str}"

    # 1. Did the agent JUST BUY? (Check this BEFORE checking if contracts > 0)
    if action_type in [1, 2] and env.contracts > 0 and prev_contracts == 0:
        opt_type = "CALL" if getattr(env, "pos_type", 1) == 1 or action_type == 1 else "PUT"
        return f"{agent_name}: BOUGHT {opt_type} @ {price_str}"

    # 2. Did the agent JUST CLOSE?
    if prev_contracts > 0 and env.contracts == 0:
        return f"{agent_name}: CLOSED POSITION @ {price_str}"

    # 3. Is the agent CONTINUING TO HOLD an open position?
    if env.contracts > 0:
        pos_label = "Call" if getattr(env, "pos_type", 1) == 1 else "Put"
        return f"{agent_name}: HELD {env.contracts} {pos_label} contract(s) @ {price_str}"

    # 4. Holding cash
    return f"{agent_name}: HELD CASH @ {price_str}"


def step_agent(npc, norm_env, env, current_price, agent_name):
    """Executes a prediction step for a single NPC and formats action output."""
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
# 4. WEBSOCKET LOOP
# -------------------------------------------------------------------

@app.websocket("/ws/stocks")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Frontend client connected!")

    # Reset environments with initial price buffers
    spy_bull_env.spy_prices = spy_recent_prices.copy()
    spy_bear_env.spy_prices = spy_recent_prices.copy()
    spy_bull_env.reset()
    spy_bear_env.reset()

    qqq_bull_env.stock_prices = qqq_recent_prices.copy()
    qqq_bear_env.stock_prices = qqq_recent_prices.copy()
    qqq_bull_env.reset()
    qqq_bear_env.reset()

    # Initialize simulation dates
    spy_sim_date = datetime.now().date()
    qqq_sim_dt = datetime.combine(datetime.now().date(), time(9, 30))

    try:
        while True:
            # --- 1. ADVANCE SIMULATION TIME (CRITICAL FIX) ---
            spy_sim_date += timedelta(days=1)
            
            # Skip weekends for SPY daily charts
            while spy_sim_date.weekday() in (5, 6):
                spy_sim_date += timedelta(days=1)

            # Advance QQQ timestamp by 5 minutes
            qqq_sim_dt = step_5m(qqq_sim_dt)

            # --- 2. STEP SIMULATORS ---
            spy_price = float(spy_sim.step_day())
            spy_recent_prices.append(spy_price)
            if len(spy_recent_prices) > 300:
                spy_recent_prices.pop(0)

            # Use intraday step method for QQQ instead of step_day()
            qqq_price = float(qqq_sim.step_5m() if hasattr(qqq_sim, 'step_5m') else qqq_sim.step_day())
            qqq_recent_prices.append(qqq_price)
            if len(qqq_recent_prices) > 300:
                qqq_recent_prices.pop(0)

            # Update env buffers
            spy_bull_env.spy_prices = spy_recent_prices.copy()
            spy_bear_env.spy_prices = spy_recent_prices.copy()
            qqq_bull_env.stock_prices = qqq_recent_prices.copy()
            qqq_bear_env.stock_prices = qqq_recent_prices.copy()

            # --- 3. STEP NPCS ---
            spy_bull_data = step_agent(spy_bull_npc, spy_bull_norm, spy_bull_env, spy_price, "SPY_Bull_NPC")
            spy_bear_data = step_agent(spy_bear_npc, spy_bear_norm, spy_bear_env, spy_price, "SPY_Bear_NPC")

            qqq_bull_data = step_agent(qqq_bull_npc, qqq_bull_norm, qqq_bull_env, qqq_price, "QQQ_Bull_NPC_5m")
            qqq_bear_data = step_agent(qqq_bear_npc, qqq_bear_norm, qqq_bear_env, qqq_price, "QQQ_Bear_NPC_5m")

            # Collect active trade logs
            active_logs = []
            all_npc_data = [
                ("SPY", spy_bull_data), ("SPY", spy_bear_data),
                ("QQQ", qqq_bull_data), ("QQQ", qqq_bear_data)
            ]

            for symbol, data in all_npc_data:
                action_str = data.get("action", "")
                if any(keyword in action_str for keyword in ["BOUGHT", "CLOSED", "LIQUIDATED", "EXPIRED"]):
                    active_logs.append({
                        "text": action_str,
                        "symbol": symbol
                    })

            # --- 4. BUILD & TRANSMIT WEBSOCKET PACKET ---
            packet = {
                "timestamp": int(time_module.time()),
                "dates": {
                    "SPY": spy_sim_date.strftime("%Y-%m-%d"),
                    "QQQ": qqq_sim_dt.strftime("%Y-%m-%d %H:%M:%S"),
                },
                "data": {
                    "SPY": round(spy_price, 2),
                    "QQQ": round(qqq_price, 2),
                },
                "npcs": {
                    "SPY": {
                        "bull": spy_bull_data,
                        "bear": spy_bear_data,
                    },
                    "QQQ": {
                        "bull": qqq_bull_data,
                        "bear": qqq_bear_data,
                    },
                },
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