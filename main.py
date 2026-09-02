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
import inspect
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from scipy.stats import norm

from helpers import SPYOptionsEnvCalls, SPYOptionsEnvPuts, SPYOptionsEnv5m, SPYStockSimulator, QQQ5mStockSimulator, HistoricalCSVStockSimulator, prepare_spy_5m_data

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
def compute_black_scholes_side(
    spot: float,
    strike: float,
    dte: float,
    is_call: bool,
    iv_rank: float = 0.35
) -> tuple[float, float]:
    """Calculates Black-Scholes bid and ask for a specific option side."""
    # 1. Effective Time to Expiration in Years
    T = max(1e-6, float(dte) / 365.0)
    
    # 2. Implied Volatility & Skew
    iv_rank = float(np.clip(iv_rank, 0.0, 1.0))
    sigma = 0.10 + (iv_rank * 0.35)
    
    # Volatility Skew Adjustment (Put Skew for OTM Puts)
    moneyness_ratio = strike / max(0.01, spot)
    if not is_call and moneyness_ratio < 1.0:
        sigma *= 1.0 + (1.0 - moneyness_ratio) * 0.8
        
    r = 0.04  # Risk-free rate (4%)

    # 3. Black-Scholes d1 & d2
    d1 = (np.log(spot / strike) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # 4. Mid Price & Delta
    if is_call:
        mid_price = spot * norm.cdf(d1) - strike * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
    else:
        mid_price = strike * np.exp(-r * T) * norm.cdf(-d2) - spot * norm.cdf(-d1)
        delta = abs(norm.cdf(d1) - 1.0)

    mid_price = max(0.01, float(mid_price))

    # 5. Market Maker Bid/Ask Spread
    otm_penalty = (1.0 - delta) * 0.02
    iv_penalty = iv_rank * 0.02
    half_spread = max(0.005, 0.005 + otm_penalty + iv_penalty)

    bid = max(0.01, round(mid_price - half_spread, 2))
    ask = max(0.02, round(mid_price + half_spread, 2))

    return bid, ask


def compute_option_row(
    spot: float,
    strike: float,
    dte: float = 7.0,
    iv_rank: float = 0.35
) -> Dict[str, Any]:
    """
    Recalculates Option Premiums using Black-Scholes while maintaining
    100% backward compatibility with the expected frontend key schema.
    """
    call_bid, call_ask = compute_black_scholes_side(spot, strike, dte, is_call=True, iv_rank=iv_rank)
    put_bid, put_ask = compute_black_scholes_side(spot, strike, dte, is_call=False, iv_rank=iv_rank)

    return {
        "strike": strike,
        "call_bid": call_bid,
        "call_ask": call_ask,
        "put_bid": put_bid,
        "put_ask": put_ask,
        "in_the_money_call": spot > strike,
        "in_the_money_put": spot < strike,
    }

# Daily Stock Simulator (SPY)
#spy_sim = SPYStockSimulator(initial_price=450.0, annual_drift=0.09, base_volatility=0.18)
#spy_recent_prices = [spy_sim.price]

# 5-Minute Intraday Stock Simulator (e.g., QQQ)
#qqq_sim = QQQStockSimulator(initial_price=380.0, base_volatility=0.22)
#qqq_recent_prices = [qqq_sim.price]

CSV_FILE = Path("spy_5m_polygon_prepared.csv")

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

# 1. Instantiate Historical CSV Simulator
historical_sim = HistoricalCSVStockSimulator(
    csv_path_or_df=MODELS_DIR / CSV_FILE,
    loop=True
)
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

def mask_fn(env):
    """Top-level mask callback function for ActionMasker."""
    return env.unwrapped.action_masks()

def load_ppo_model_safe(model_file: str, env: Any) -> PPO:
    model_path = MODELS_DIR / model_file

    # Define policy network architecture matching your training configuration
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=torch.nn.Tanh,
        ortho_init=True,
    )

    # 1. Try native Stable-Baselines3 load first
    try:
        model = PPO("MlpPolicy", env, device="cpu")
        with zipfile.ZipFile(model_path, "r") as z:
            policy_bytes = z.read("policy.pth")
    
        buffer = io.BytesIO(policy_bytes)
        state_dict = torch.load(buffer, map_location="cpu", weights_only=False)
        model.policy.load_state_dict(state_dict)
        return model
    except Exception as e:
        print(f"Native PPO.load failed for {model_file}, attempting MaskablePPO.load")

    # 2. Fallback Attempt: Construct matching PPO instance and extract policy weights from zip
    model = MaskablePPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        device="cpu",
    )

    # Read state dict directly from the saved model zip
    with zipfile.ZipFile(model_path, "r") as z:
        policy_bytes = z.read("policy.pth")

    buffer = io.BytesIO(policy_bytes)
    state_dict = torch.load(buffer, map_location="cpu", weights_only=False)
    model.policy.load_state_dict(state_dict)
    return model


# -------------------------------------------------------------------
# 2. MODULAR STOCK REGISTRY ARCHITECTURE
# -------------------------------------------------------------------
@dataclass
class StockContext:
    symbol: str
    interval: str  # "1d" or "5m"
    simulator: Any
    bull_env: Any
    bear_env: Any
    bull_npc: Any  # PPO
    bull_norm: Optional[Any]  # Optional[VecNormalize]
    bear_npc: Any  # PPO
    bear_norm: Optional[Any]  # Optional[VecNormalize]
    current_dt: datetime
    initial_dt: datetime
    recent_prices: List[float] = field(default_factory=list)
    recent_vwap_dist: List[float] = field(default_factory=list)
    recent_iv_rank: List[float] = field(default_factory=list)

    # Added feature buffers for the 5-minute model
    recent_rsi: List[float] = field(default_factory=list)
    recent_sma_ratio: List[float] = field(default_factory=list)
    recent_z_ret_1bar: List[float] = field(default_factory=list)
    recent_z_ret_6bar: List[float] = field(default_factory=list)
    recent_vol_expansion_ratio: List[float] = field(default_factory=list)
    recent_sin_time: List[float] = field(default_factory=list)
    recent_cos_time: List[float] = field(default_factory=list)

    def _sync_env_buffers(self, env: Any):
        """Pushes rolling price and feature buffers into 1d or 5m environments cleanly."""
        unwrapped_env = env.unwrapped if hasattr(env, "unwrapped") else (env.envs[0] if hasattr(env, "envs") else env)

        prices_list = self.recent_prices.copy()
        prices_arr = np.array(prices_list, dtype=np.float32)
        n_buffer = len(prices_list)

        # Generate rolling 5m timestamps ending at self.current_dt
        if self.interval == "5m":
            timestamps = [self.current_dt - timedelta(minutes=5 * (n_buffer - 1 - i)) for i in range(n_buffer)]
        else:
            timestamps = [self.current_dt - timedelta(days=(n_buffer - 1 - i)) for i in range(n_buffer)]

        # Standard price updates
        if hasattr(unwrapped_env, "spy_prices"):
            unwrapped_env.spy_prices = prices_list
        if hasattr(unwrapped_env, "raw_spy_prices"):
            unwrapped_env.raw_spy_prices = prices_list.copy()
        if hasattr(unwrapped_env, "stock_prices"):
            unwrapped_env.stock_prices = prices_list.copy()

        if hasattr(unwrapped_env, "df") and isinstance(unwrapped_env.df, pd.DataFrame):
            n_rows = len(unwrapped_env.df)

            data_dict = {
                "timestamp": timestamps,
                "close": prices_arr,
                "high": prices_arr * 1.0005,
                "low": prices_arr * 0.9995,
                "volume": [1000.0] * n_buffer,
                "vwap_dist": np.array(self.recent_vwap_dist, dtype=np.float32),
                "iv_rank": np.array(self.recent_iv_rank, dtype=np.float32),
                "rsi": np.array(self.recent_rsi, dtype=np.float32),
                "sma_ratio": np.array(self.recent_sma_ratio, dtype=np.float32),
                "z_ret_1bar": np.array(self.recent_z_ret_1bar, dtype=np.float32),
                "z_ret_6bar": np.array(self.recent_z_ret_6bar, dtype=np.float32),
                "vol_expansion_ratio": np.array(self.recent_vol_expansion_ratio, dtype=np.float32),
                "sin_time": np.array(self.recent_sin_time, dtype=np.float32),
                "cos_time": np.array(self.recent_cos_time, dtype=np.float32),
            }

            if n_rows != n_buffer:
                for col in unwrapped_env.df.columns:
                    if col not in data_dict:
                        data_dict[col] = (
                            unwrapped_env.df[col].values[-n_buffer:]
                            if len(unwrapped_env.df[col]) >= n_buffer
                            else [0.0] * n_buffer
                        )
                unwrapped_env.df = pd.DataFrame(data_dict)
            else:
                for k, v in data_dict.items():
                    if k in unwrapped_env.df.columns:
                        unwrapped_env.df[k] = v

            unwrapped_env.raw_spy_prices = (
                unwrapped_env.df["close"].values.astype(np.float32).tolist()
            )
            unwrapped_env.spy_prices = unwrapped_env.raw_spy_prices.copy()

            if hasattr(unwrapped_env, "price"):
                unwrapped_env.price = prices_list[-1]

            latest_idx = max(0, len(unwrapped_env.df) - 1)
            if hasattr(unwrapped_env, "data_idx"):
                unwrapped_env.data_idx = latest_idx
            if hasattr(unwrapped_env, "frame_idx"):
                unwrapped_env.frame_idx = latest_idx

    def reset_state(self):
        """Resets environments and ensures historical buffers are preserved at step index."""
        # 1. Reset base environments first
        self.bull_env.reset()
        self.bear_env.reset()

        # 2. Re-apply synced historical buffers after reset
        self._sync_env_buffers(self.bull_env)
        self._sync_env_buffers(self.bear_env)

        # 3. Advance environment internal step pointer past warm-up window if present
        for env in (self.bull_env, self.bear_env):
            if hasattr(env, "current_step"):
                # Set cursor to end of historical buffer so model trades immediately
                env.current_step = len(self.recent_prices) - 1
            if hasattr(env, "frame_idx"):
                env.frame_idx = len(self.recent_prices) - 1

        self.current_dt = self.initial_dt

    def step(self) -> float:
        """Ticks simulator and advances timestamp regardless of active UI tab."""
        # 1. Advance Datetime State
        if self.interval == "5m":
            self.current_dt = step_5m(self.current_dt)
        else:
            self.current_dt = step_1d(self.current_dt)

        # 2. Extract Price & Technical Features from Simulator
        if hasattr(self.simulator, "step_bar"):
            res = self.simulator.step_bar()
        elif hasattr(self.simulator, "step_5m") and self.interval == "5m":
            res = self.simulator.step_5m()
        elif hasattr(self.simulator, "step_day"):
            res = self.simulator.step_day()
        else:
            res = getattr(self.simulator, "price", 0.0)

        if isinstance(res, dict):
            price = float(res.get("close", res.get("price", 0.0)))
            vwap_dist = float(res.get("vwap_dist", 0.0))
            iv_rank = float(res.get("iv_rank", 0.5))
            rsi = float(res.get("rsi", 50.0))
            sma_ratio = float(res.get("sma_ratio", 0.0))
            z_ret_1bar = float(res.get("z_ret_1bar", 0.0))
            z_ret_6bar = float(res.get("z_ret_6bar", 0.0))
            vol_expansion_ratio = float(res.get("vol_expansion_ratio", 0.0))
            sin_time = float(res.get("sin_time", 0.0))
            cos_time = float(res.get("cos_time", 1.0))
        else:
            price = float(res)
            vwap_dist, iv_rank, rsi, sma_ratio = 0.0, 0.5, 50.0, 0.0
            z_ret_1bar, z_ret_6bar, vol_expansion_ratio = 0.0, 0.0, 0.0
            sin_time, cos_time = 0.0, 1.0

        # 3. Update Rolling Windows (300-bar max)
        self.recent_prices.append(price)
        self.recent_vwap_dist.append(vwap_dist)
        self.recent_iv_rank.append(iv_rank)
        self.recent_rsi.append(rsi)
        self.recent_sma_ratio.append(sma_ratio)
        self.recent_z_ret_1bar.append(z_ret_1bar)
        self.recent_z_ret_6bar.append(z_ret_6bar)
        self.recent_vol_expansion_ratio.append(vol_expansion_ratio)
        self.recent_sin_time.append(sin_time)
        self.recent_cos_time.append(cos_time)

        if len(self.recent_prices) > 300:
            self.recent_prices.pop(0)
            self.recent_vwap_dist.pop(0)
            self.recent_iv_rank.pop(0)
            self.recent_rsi.pop(0)
            self.recent_sma_ratio.pop(0)
            self.recent_z_ret_1bar.pop(0)
            self.recent_z_ret_6bar.pop(0)
            self.recent_vol_expansion_ratio.pop(0)
            self.recent_sin_time.pop(0)
            self.recent_cos_time.pop(0)

        # 4. Sync updated prices and dataframes into environments
        self._sync_env_buffers(self.bull_env)
        self._sync_env_buffers(self.bear_env)

        return price

    def get_formatted_time(self) -> str:
        if self.current_dt.tzinfo is None:
            utc_dt = self.current_dt.replace(tzinfo=zoneinfo.ZoneInfo("UTC"))
        else:
            utc_dt = self.current_dt

        eastern_dt = utc_dt.astimezone(zoneinfo.ZoneInfo("America/New_York"))

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
    bear_model_file: str,
    bull_norm_file: Optional[str] = None,
    bear_norm_file: Optional[str] = None,
) -> StockContext:
    """Factory builder for registering any stock model into runtime context."""

    prices_buffer = []
    vwap_buffer = []
    iv_buffer = []
    rsi_buffer = []
    sma_ratio_buffer = []
    z_ret_1bar_buffer = []
    z_ret_6bar_buffer = []
    vol_expansion_buffer = []
    sin_time_buffer = []
    cos_time_buffer = []
    high_prices = []
    low_prices = []
    volume_buffer = []
    timestamp_buffer = []

    # Base start date for synthetic generation fallbacks
    base_start_dt = datetime.now() - timedelta(minutes=5 * 300)

    # 1. Populate buffers according to Simulator Type
    if hasattr(simulator, "df") and isinstance(getattr(simulator, "df"), pd.DataFrame) and len(simulator.df) >= 300:
        historical_slice = simulator.df.iloc[:300]
        prices_buffer = historical_slice["close"].astype(float).tolist()
        high_prices = historical_slice["high"].astype(float).tolist() if "high" in historical_slice.columns else [p * 1.001 for p in prices_buffer]
        low_prices = historical_slice["low"].astype(float).tolist() if "low" in historical_slice.columns else [p * 0.999 for p in prices_buffer]
        volume_buffer = historical_slice["volume"].astype(float).tolist() if "volume" in historical_slice.columns else [1000.0] * 300
        vwap_buffer = historical_slice["vwap_dist"].astype(float).tolist() if "vwap_dist" in historical_slice.columns else [0.0] * 300
        iv_buffer = historical_slice["iv_rank"].astype(float).tolist() if "iv_rank" in historical_slice.columns else [0.5] * 300
        rsi_buffer = historical_slice["rsi"].astype(float).tolist() if "rsi" in historical_slice.columns else [50.0] * 300
        sma_ratio_buffer = historical_slice["sma_ratio"].astype(float).tolist() if "sma_ratio" in historical_slice.columns else [0.0] * 300
        z_ret_1bar_buffer = historical_slice["z_ret_1bar"].astype(float).tolist() if "z_ret_1bar" in historical_slice.columns else [0.0] * 300
        z_ret_6bar_buffer = historical_slice["z_ret_6bar"].astype(float).tolist() if "z_ret_6bar" in historical_slice.columns else [0.0] * 300
        vol_expansion_buffer = historical_slice["vol_expansion_ratio"].astype(float).tolist() if "vol_expansion_ratio" in historical_slice.columns else [0.0] * 300
        sin_time_buffer = historical_slice["sin_time"].astype(float).tolist() if "sin_time" in historical_slice.columns else [0.0] * 300
        cos_time_buffer = historical_slice["cos_time"].astype(float).tolist() if "cos_time" in historical_slice.columns else [1.0] * 300
        if "timestamp" in historical_slice.columns:
            timestamp_buffer = pd.to_datetime(historical_slice["timestamp"]).tolist()
        else:
            timestamp_buffer = [base_start_dt + timedelta(minutes=5 * i) for i in range(300)]
        # Fast-forward simulator index so live steps start at index 300
        if hasattr(simulator, "current_idx"):
            simulator.current_idx = 300

    elif hasattr(simulator, "step_bar") or hasattr(simulator, "step_5m"):
        step_fn = getattr(simulator, "step_bar", getattr(simulator, "step_5m", None))
        for i in range(300):
            bar = step_fn()
            prices_buffer.append(float(bar["close"]))
            high_prices.append(float(bar["high"]))
            low_prices.append(float(bar["low"]))
            volume_buffer.append(float(bar["volume"]))
            vwap_buffer.append(float(bar.get("vwap_dist", 0.0)))
            iv_buffer.append(float(bar.get("iv_rank", 0.5)))
            rsi_buffer.append(float(bar.get("rsi", 50.0)))
            sma_ratio_buffer.append(float(bar.get("sma_ratio", 0.0)))
            z_ret_1bar_buffer.append(float(bar.get("z_ret_1bar", 0.0)))
            z_ret_6bar_buffer.append(float(bar.get("z_ret_6bar", 0.0)))
            vol_expansion_buffer.append(float(bar.get("vol_expansion_ratio", 0.0)))
            sin_time_buffer.append(float(bar.get("sin_time", 0.0)))
            cos_time_buffer.append(float(bar.get("cos_time", 1.0)))
            ts = bar.get("timestamp", base_start_dt + timedelta(minutes=5 * i))
            timestamp_buffer.append(ts)

        # Position simulator to index 300 rather than looping back to 0
        if hasattr(simulator, "reset"):
            try:
                simulator.reset(start_idx=300)
            except TypeError:
                simulator.reset()

    elif hasattr(simulator, "step_day"):
        # Pre-calculate 300 daily timestamps if the simulator lacks a dynamic `current_dt`
        fallback_dates = pd.bdate_range(end=base_start_dt, periods=300).to_pydatetime().tolist()
        
        for i in range(300):
            p = float(simulator.step_day())
            prices_buffer.append(p)
            high_prices.append(p * 1.001)
            low_prices.append(p * 0.999)
            volume_buffer.append(1000.0)
            vwap_buffer.append(0.0)
            iv_buffer.append(0.5)
            rsi_buffer.append(50.0)
            sma_ratio_buffer.append(0.0)
            z_ret_1bar_buffer.append(0.0)
            z_ret_6bar_buffer.append(0.0)
            vol_expansion_buffer.append(0.0)
            sin_time_buffer.append(0.0)
            cos_time_buffer.append(1.0)
            
            # Capture updated current_dt from step_day(), or use calculated fallback date
            ts = getattr(simulator, "current_dt", fallback_dates[i])
            timestamp_buffer.append(ts)

        if hasattr(simulator, "reset"):
            simulator.reset()

    if prices_buffer:
        simulator.price = prices_buffer[-1]

    # 2. Build complete seed DataFrame with all features
    seed_df = pd.DataFrame(
        {
            "timestamp": timestamp_buffer,
            "close": prices_buffer,
            "high": high_prices,
            "low": low_prices,
            "volume": volume_buffer,
            "vwap_dist": vwap_buffer,
            "iv_rank": iv_buffer,
            "rsi": rsi_buffer,
            "sma_ratio": sma_ratio_buffer,
            "z_ret_1bar": z_ret_1bar_buffer,
            "z_ret_6bar": z_ret_6bar_buffer,
            "vol_expansion_ratio": vol_expansion_buffer,
            "sin_time": sin_time_buffer,
            "cos_time": cos_time_buffer,
        }
    )

    # 3. Environment Instantiation & Setup
    def instantiate_env(env_cls):
        sig = inspect.signature(env_cls.__init__)
        params = sig.parameters

        kwargs = {}
        if "is_eval" in params:
            kwargs["is_eval"] = True
        if "train" in params:
            kwargs["train"] = False

        if "df_5m" in params:
            kwargs["df_5m"] = seed_df.copy()

        if "spy_prices" in params:
            kwargs["spy_prices"] = prices_buffer.copy()
        elif "stock_prices" in params:
            kwargs["stock_prices"] = prices_buffer.copy()

        env = env_cls(**kwargs)
        
        # Fast-forward internal pointer if environment uses a lookback index
        if hasattr(env, "current_step") and len(seed_df) > 0:
            env.current_step = len(seed_df) - 1

        if "df_5m" in params:
            masked_env = ActionMasker(env, mask_fn)
            return masked_env
        
        return env

    bull_env = instantiate_env(bull_env_cls)
    bear_env = instantiate_env(bear_env_cls)

    bull_vec_env = DummyVecEnv([lambda: bull_env])
    bear_vec_env = DummyVecEnv([lambda: bear_env])

    if bull_norm_file:
        bull_norm = VecNormalize.load(
            str(MODELS_DIR / bull_norm_file), bull_vec_env
        )
        bull_norm.training = False
        bull_eval_env = bull_norm
    else:
        bull_norm = None
        bull_eval_env = bull_vec_env

    bull_npc = load_ppo_model_safe(bull_model_file, bull_eval_env)

    if bear_norm_file:
        bear_norm = VecNormalize.load(
            str(MODELS_DIR / bear_norm_file), bear_vec_env
        )
        bear_norm.training = False
        bear_eval_env = bear_norm
    else:
        bear_norm = None
        bear_eval_env = bear_vec_env

    bear_npc = load_ppo_model_safe(bear_model_file, bear_eval_env)

    init_dt = (
        timestamp_buffer[-1]
        if timestamp_buffer
        else datetime.combine(
            datetime.now().date() - timedelta(days=300 if interval == "1d" else 0),
            time(9, 30) if interval == "5m" else time(0, 0),
        )
    )

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
        initial_dt=init_dt,
        recent_prices=prices_buffer,
        recent_vwap_dist=vwap_buffer,
        recent_iv_rank=iv_buffer,
        recent_rsi=rsi_buffer,
        recent_sma_ratio=sma_ratio_buffer,
        recent_z_ret_1bar=z_ret_1bar_buffer,
        recent_z_ret_6bar=z_ret_6bar_buffer,
        recent_vol_expansion_ratio=vol_expansion_buffer,
        recent_sin_time=sin_time_buffer,
        recent_cos_time=cos_time_buffer,
    )

# --- GLOBAL STOCKS REGISTRY ---
STOCKS: Dict[str, StockContext] = {
    "SPY": build_stock_context(
        symbol="SPY",
        interval="1d",
        simulator=SPYStockSimulator(
            initial_price=450.0, annual_drift=0.09, base_volatility=0.18
        ),
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
        simulator=QQQ5mStockSimulator(initial_price=500.0, annual_drift=0.08),
        bull_env_cls=SPYOptionsEnv5m,
        bear_env_cls=SPYOptionsEnv5m,
        bull_model_file="best_model11p(120).zip",
        bear_model_file="best_model10p(90).zip",
    ),
    "SPO": build_stock_context(
        symbol="SPO",
        interval="5m",
        simulator=historical_sim,
        bull_env_cls=SPYOptionsEnv5m,
        bear_env_cls=SPYOptionsEnv5m,
        bull_model_file="best_model11p(120).zip",
        bear_model_file="best_model10p(90).zip",
    )
}


# -------------------------------------------------------------------
# 3. AGENT STEPPING HELPERS
# -------------------------------------------------------------------

def parse_npc_action(agent_name: str, action_vec: Any, current_price: float, env: Any, prev_contracts: int = 0) -> str:
    action_type = int(action_vec[0]) if isinstance(action_vec, (list, tuple, np.ndarray)) else int(action_vec)
    price_str = f"${current_price:.2f}"
    #print(env.contracts)
    last_event = getattr(env, "closed", None)
    #print(last_event)
    if last_event == "Trailing Stop" or last_event == "Hard Stop":
        return f"{agent_name}: LIQUIDATED (Stop-Loss Hit) @ {price_str}"
    elif last_event == "Expired":
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

def step_agent(
    npc: PPO,
    norm_env: Optional[VecNormalize],
    env: Any,
    current_price: float,
    agent_name: str,
) -> dict:
    base_env = env.unwrapped if hasattr(env, "unwrapped") else (env.envs[0] if hasattr(env, "envs") else env)

    # 1. Fetch Observation
    if hasattr(env, "_get_normalized_observation"):
        raw_obs = base_env._get_normalized_observation()
    elif hasattr(base_env, "_get_obs"):
        raw_obs = base_env._get_obs()
    elif hasattr(base_env, "get_obs"):
        raw_obs = base_env.get_obs()
    else:
        raw_obs = getattr(base_env, "observation", np.zeros(10))

    raw_obs = np.asarray(raw_obs, dtype=np.float32).flatten()
    obs_batch = np.expand_dims(raw_obs, axis=0)

    # 2. Normalize Observation
    if norm_env is not None:
        norm_env.training = False
        norm_obs = norm_env.normalize_obs(obs_batch)
    else:
        norm_obs = obs_batch

    # if isinstance(npc, MaskablePPO):
    #     row = base_env.df.iloc[base_env.data_idx]
    #     print("vwap_dist: ", float(row.get("vwap_dist", 0.0)), "iv_rank: ", float(row.get("iv_rank", 0.0)))
    #      print("cash: ", base_env.cash, "timestamp: ", row.get("timestamp", None))

    # 3. Predict Action
    if isinstance(npc, MaskablePPO):
        action_masks = env.action_masks()
        obs = env.unwrapped._get_normalized_observation()
        action, _states = npc.predict(obs, action_masks=action_masks, deterministic=True)
    else:
        action, _states = npc.predict(norm_obs, deterministic=True)
    act_vec = action[0] if isinstance(action, np.ndarray) and action.ndim > 1 else action

    # 4. CAPTURE CONTRACTS BEFORE STEP (0 or 1)
    prev_contracts = int(getattr(base_env, "contracts", 0))

    # 5. Execute Step (Environment updates base_env.contracts here)
    step_res = env.step(act_vec)
    done = step_res[2] if len(step_res) == 4 else (step_res[2] or step_res[3])

    #print(f"[{agent_name}] Action: {act_vec} | Prev: {prev_contracts} | New: {base_env.contracts}")
    # Debug trace when model attempts to buy but contracts stay 0
    if act_vec[0] in (1, 2) and prev_contracts == 0 and base_env.contracts == 0:
        cash = getattr(base_env, "cash", 0.0)
        curr_price = getattr(base_env, "price", 0.0)
        entry_price = getattr(base_env, "entry_price", 0.0)
        print(f"⚠️ [{agent_name}] BUY ORDER FAILED | Cash: ${cash:.2f} | Price: ${curr_price:.2f} | Entry Price: ${entry_price:.2f}")
    # 6. Parse Action Log (Compares prev_contracts against new base_env.contracts)
    action_log = parse_npc_action(
        agent_name,
        act_vec,
        current_price,
        base_env,
        prev_contracts=prev_contracts,
    )

    # 7. Soft reset on done
    if done:
        if hasattr(base_env, "close_position") and getattr(base_env, "contracts", 0) > 0:
            try:
                base_env.close_position(current_price)
            except Exception:
                pass

        if hasattr(base_env, "_reset_position_state"):
            base_env._reset_position_state()

    # 8. Calculate Portfolio Value
    if hasattr(base_env, "_get_portfolio_value"):
        portfolio_val = base_env._get_portfolio_value(current_price)
    elif hasattr(base_env, "cash") and hasattr(base_env, "contracts"):
        portfolio_val = base_env.cash + (base_env.contracts * current_price * 100.0)
    else:
        portfolio_val = getattr(base_env, "portfolio_value", 10000.0)

    return {
        "action": str(action_log),
        "portfolio_value": round(float(portfolio_val), 2),
        "cash": round(float(getattr(base_env, "cash", 0.0)), 2),
        "contracts": int(getattr(base_env, "contracts", 0)),
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

    # Fetch current IV rank from context if available, otherwise default to 0.35
    ctx = STOCKS.get(symbol)
    env = getattr(ctx, "bull_env", None)
    base_env = env.unwrapped if hasattr(env, "unwrapped") else (env.envs[0] if hasattr(env, "envs") else env)
    iv_rank = getattr(ctx, "current_iv_rank", 0.35) if ctx else 0.35
    if hasattr(base_env, "is_inverted"):
        row = base_env.df.iloc[base_env.data_idx]
        iv_rank = float(row["iv_rank"])
    # Dynamic strike spacing
    strike_step = 2.5 if current_price < 500 else 5.0
    center_strike = round(current_price / strike_step) * strike_step
    strikes = [round(center_strike + (i * strike_step), 2) for i in range(-5, 6)]

    chain = [
        compute_option_row(spot=current_price, strike=strike, dte=dte, iv_rank=iv_rank)
        for strike in strikes
    ]
    return {"symbol": symbol, "chain": chain}


# -------------------------------------------------------------------
# 5. WEBSOCKET LOOP (PARALLEL TICKING FOR ALL STOCKS)
# -------------------------------------------------------------------

def sanitize_payload(obj):
    """Recursively casts NumPy/Pandas data types into native JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {k: sanitize_payload(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_payload(v) for v in obj]
    elif isinstance(obj, (np.floating, np.complexfloating)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.signedinteger, np.unsignedinteger)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (pd.Timestamp, datetime)):
        return str(obj)
    elif pd.isna(obj):
        return None
    return obj


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
                bull_data = step_agent(
                    ctx.bull_npc,
                    ctx.bull_norm,
                    ctx.bull_env,
                    price,
                    f"{symbol}_Bull_NPC",
                )
                bear_data = step_agent(
                    ctx.bear_npc,
                    ctx.bear_norm,
                    ctx.bear_env,
                    price,
                    f"{symbol}_Bear_NPC",
                )

                # Record Payloads
                dates_payload[symbol] = ctx.get_formatted_time()
                data_payload[symbol] = (
                    round(float(price), 2) if price is not None else 0.0
                )
                npcs_payload[symbol] = {
                    "bull": bull_data,
                    "bear": bear_data,
                }

                # Collect Active Trade Logs across all stocks
                for data in [bull_data, bear_data]:
                    action_str = str(data.get("action", ""))
                    #print(action_str)
                    if any(
                        kw in action_str
                        for kw in [
                            "BOUGHT",
                            "CLOSED",
                            "LIQUIDATED",
                            "EXPIRED",
                            "HELD",
                        ]
                    ):
                        active_logs.append(
                            {"text": action_str, "symbol": symbol}
                        )

            # Consolidated Packet for ALL symbols
            packet = {
                "timestamp": int(time_module.time()),
                "dates": dates_payload,
                "data": data_payload,
                "npcs": npcs_payload,
                "logs": active_logs,
            }

            # Sanitize numpy types to prevent JSON serialization crash on step 2
            safe_packet = sanitize_payload(packet)

            await websocket.send_json(safe_packet)
            await asyncio.sleep(3)

    except WebSocketDisconnect:
        print("Frontend client disconnected cleanly.")
    except Exception as e:
        print(f"❌ Error in WebSocket loop: {e}")
        import traceback

        traceback.print_exc()
        try:
            await websocket.close()
        except Exception:
            pass  # Socket was already closed or broken