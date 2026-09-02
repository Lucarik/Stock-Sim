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
import collections

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

from helpers import SPYOptionsEnvCalls, SPYOptionsEnvPuts, SPYOptionsEnv5m, prepare_spy_5m_data

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

class QQQ5mStockSimulator:
    """Simulates realistic 5-minute SPY/QQQ intraday bar data with regime-dependent jump shocks.

    Features:
    - Dynamic Regime Jumps: Jump probability and severity expand during high-volatility regimes.
    - Intraday U-shaped volatility & volume profile.
    - GARCH-like stochastic volatility clustering.
    - Direct compatibility with StockContext: Returns close, vwap_dist, and iv_rank dynamically.
    """

    def __init__(
        self,
        initial_price: float = 500.0,
        annual_drift: float = 0.08,
        base_volatility: float = 0.16,
        average_daily_volume: int = 50_000_000,
        base_jump_prob: float = 0.0005,
        max_history: int = 300,
    ):
        self._price = float(initial_price)
        self.initial_price = float(initial_price)
        self.max_history = max_history

        # Time step parameters: 252 trading days * 78 five-minute bars = 19,656 bars/year
        self.bars_per_day = 78
        self.dt = 1.0 / (252.0 * self.bars_per_day)

        # Drift and Volatility (annualized base)
        self.mu_base = annual_drift
        self.sigma_base = base_volatility
        self.current_sigma = base_volatility

        # Volatility persistence per 5m step
        self.vol_persistence = 0.998

        # Volume parameters
        self.avg_daily_vol = average_daily_volume
        self.avg_bar_vol = self.avg_daily_vol / self.bars_per_day

        # Baseline Jump Parameters
        self.base_jump_prob = base_jump_prob
        self.base_jump_mean = -0.002
        self.base_jump_std = 0.008

        # Tracking intraday bar state
        self.bar_index_in_day = 0

        # Technical Feature State Tracking (for VWAP and Parkinson Volatility / IV Rank)
        self.highs = collections.deque([self._price], maxlen=max_history)
        self.lows = collections.deque([self._price], maxlen=max_history)
        self.cum_pv = self._price * 1000.0
        self.cum_vol = 1000.0

    @property
    def price(self) -> float:
        """Property fallback for StockContext price extraction."""
        return self._price

    @price.setter
    def price(self, val: float):
        self._price = float(val)

    def _get_intraday_multipliers(self, step_in_day: int) -> tuple[float, float]:
        """Computes U-shaped volatility and volume curves for market open/close dynamics."""
        x = step_in_day / (self.bars_per_day - 1)
        vol_mult = 1.8 - 2.8 * x + 2.8 * (x**2)
        volu_mult = 2.5 - 4.2 * x + 4.2 * (x**2)
        return max(0.4, vol_mult), max(0.2, volu_mult)

    def _compute_regime_jump_parameters(self) -> tuple[float, float, float]:
        """Scales jump probability, mean drop size, and shock variance based on current volatility regime."""
        vol_ratio = self.current_sigma / self.sigma_base
        effective_jump_prob = min(0.02, self.base_jump_prob * (vol_ratio**2))
        effective_jump_mean = self.base_jump_mean * (vol_ratio**1.3)
        effective_jump_std = self.base_jump_std * vol_ratio
        return effective_jump_prob, effective_jump_mean, effective_jump_std

    def _calculate_features(
        self, price: float, high: float, low: float, volume: float
    ) -> tuple[float, float]:
        """Calculates running VWAP distance and IV Rank proxy."""
        # 1. Update Intraday VWAP
        typical_price = (high + low + price) / 3.0
        self.cum_pv += typical_price * volume
        self.cum_vol += volume

        current_vwap = (
            self.cum_pv / self.cum_vol if self.cum_vol > 0 else price
        )
        vwap_dist = float((price - current_vwap) / current_vwap)

        # 2. Parkinson Volatility Proxy for IV Rank (rolling window)
        self.highs.append(high)
        self.lows.append(low)

        if len(self.highs) > 5:
            highs_arr = np.array(self.highs, dtype=np.float32)
            lows_arr = np.array(self.lows, dtype=np.float32)
            ratio = np.maximum(highs_arr / np.maximum(lows_arr, 1e-5), 1.0)
            log_hl = np.log(ratio) ** 2
            parkinson_vol = np.sqrt(
                (1.0 / (4.0 * np.log(2.0) * len(highs_arr))) * np.sum(log_hl)
            )
            iv_rank = float(np.clip((parkinson_vol - 0.005) / 0.025, 0.0, 1.0))
        else:
            iv_rank = 0.5

        return vwap_dist, iv_rank

    def step_bar(self) -> dict[str, float]:
        """Advances the simulator by 1 five-minute bar and returns metrics."""
        # Reset VWAP tracking at the start of a new trading day
        if self.bar_index_in_day == 0:
            self.cum_pv = 0.0
            self.cum_vol = 0.0

        vol_mult, volu_mult = self._get_intraday_multipliers(
            self.bar_index_in_day
        )

        # 1. Volatility Clustering
        vol_shock = np.random.normal(0, 1)
        effective_sigma_base = self.sigma_base * vol_mult
        self.current_sigma = max(
            0.04,
            self.vol_persistence * self.current_sigma
            + (1.0 - self.vol_persistence) * effective_sigma_base
            + 0.0035 * abs(vol_shock),
        )

        # 2. Jump Dynamics
        jump_prob, jump_mean, jump_std = self._compute_regime_jump_parameters()

        # 3. Drift & Diffusion
        deviation_from_base = (
            self._price - self.initial_price
        ) / self.initial_price
        trend_pull = -0.05 * deviation_from_base
        effective_mu = self.mu_base + trend_pull

        epsilon = np.random.normal(0, 1)
        drift = (effective_mu - 0.5 * (self.current_sigma**2)) * self.dt
        diffusion = self.current_sigma * np.sqrt(self.dt) * epsilon

        jump = 0.0
        if np.random.rand() < jump_prob:
            jump = np.random.normal(jump_mean, jump_std)
            self.current_sigma += abs(jump) * 2.0

        # 4. Price & Range Synthesis
        open_price = self._price
        bar_return = np.exp(drift + diffusion + jump)
        close_price = max(0.01, open_price * bar_return)

        bar_sigma = self.current_sigma * np.sqrt(self.dt)
        intrabar_noise_h = abs(np.random.normal(0, bar_sigma * 0.6))
        intrabar_noise_l = abs(np.random.normal(0, bar_sigma * 0.6))

        high_price = max(open_price, close_price) * (1.0 + intrabar_noise_h)
        low_price = min(open_price, close_price) * (1.0 - intrabar_noise_l)

        # 5. Volume Synthesis
        price_change_pct = abs(close_price - open_price) / open_price
        volume_noise = np.random.lognormal(mean=0, sigma=0.25)
        bar_volume = float(
            max(
                100,
                int(
                    self.avg_bar_vol
                    * volu_mult
                    * (1.0 + 60.0 * price_change_pct)
                    * volume_noise
                ),
            )
        )

        # Update State
        self._price = float(close_price)
        self.bar_index_in_day = (
            self.bar_index_in_day + 1
        ) % self.bars_per_day

        # Compute Technical Features
        vwap_dist, iv_rank = self._calculate_features(
            close_price, high_price, low_price, bar_volume
        )

        return {
            "open": round(open_price, 2),
            "high": round(high_price, 2),
            "low": round(low_price, 2),
            "close": round(close_price, 2),
            "volume": bar_volume,
            "vwap_dist": vwap_dist,
            "iv_rank": iv_rank,
        }

    def step_5m(self) -> dict[str, float]:
        """Explicit alias for StockContext 5-minute interval stepping."""
        return self.step_bar()

    def generate_5m_dataframe(
        self, num_days: int = 60, start_date: datetime = None
    ) -> pd.DataFrame:
        """Generates a multi-day 5-minute DataFrame formatted for prepare_spy_5m_data."""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=int(num_days * 1.5))

        records = []
        current_curr_date = start_date.date()
        days_created = 0

        while days_created < num_days:
            if current_curr_date.weekday() >= 5:
                current_curr_date += timedelta(days=1)
                continue

            market_open_dt = datetime.combine(current_curr_date, time(9, 30))
            self.bar_index_in_day = 0

            for i in range(self.bars_per_day):
                bar_time = market_open_dt + timedelta(minutes=5 * i)
                bar_data = self.step_bar()
                bar_data["timestamp"] = bar_time
                records.append(bar_data)

            days_created += 1
            current_curr_date += timedelta(days=1)

        df = pd.DataFrame(records)
        return df[
            [
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "vwap_dist",
                "iv_rank",
            ]
        ]

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

    def _sync_env_buffers(self, env: Any):
        """Pushes rolling price and feature buffers into 1d or 5m environments."""
        prices_list = self.recent_prices.copy()
        prices_arr = np.array(prices_list, dtype=np.float32)

        # Standard price updates (1d model compatibility)
        if hasattr(env, "spy_prices"):
            env.spy_prices = prices_list
        if hasattr(env, "raw_spy_prices"):
            env.raw_spy_prices = prices_list.copy()
        if hasattr(env, "stock_prices"):
            env.stock_prices = prices_list.copy()

        # Update pandas DataFrame for 5m models initialized with df_5m
        if hasattr(env, "df") and isinstance(env.df, pd.DataFrame):
            n_rows = len(env.df)
            n_buffer = len(prices_list)

            # Keep DataFrame length matching current rolling buffer length
            if n_rows != n_buffer:
                env.df = pd.DataFrame(
                    {
                        "close": prices_arr,
                        "high": prices_arr,
                        "low": prices_arr,
                        "volume": [1000.0] * n_buffer,
                        "vwap_dist": np.array(self.recent_vwap_dist, dtype=np.float32),
                        "iv_rank": np.array(self.recent_iv_rank, dtype=np.float32),
                    }
                )
            else:
                env.df["close"] = prices_arr
                if "vwap_dist" in env.df.columns:
                    env.df["vwap_dist"] = np.array(
                        self.recent_vwap_dist, dtype=np.float32
                    )
                if "iv_rank" in env.df.columns:
                    env.df["iv_rank"] = np.array(
                        self.recent_iv_rank, dtype=np.float32
                    )

            # Keep env raw price lists perfectly in sync with internal df
            env.raw_spy_prices = env.df["close"].values.astype(np.float32).tolist()
            env.spy_prices = env.raw_spy_prices.copy()

    def reset_state(self):
        """Resets buffers and restores initial state."""
        init_price = float(getattr(self.simulator, "price", 100.0))
        self.recent_prices = [init_price] * 300
        self.recent_vwap_dist = [0.0] * 300
        self.recent_iv_rank = [0.5] * 300

        self._sync_env_buffers(self.bull_env)
        self._sync_env_buffers(self.bear_env)

        self.bull_env.reset()
        self.bear_env.reset()

        self.current_dt = self.initial_dt

    def step(self) -> float:
        """Ticks simulator and advances timestamp regardless of active UI tab."""
        # 1. Advance Datetime State
        if self.interval == "5m":
            self.current_dt = step_5m(self.current_dt)
        else:
            self.current_dt = step_1d(self.current_dt)

        # 2. Extract Price & Technical Features from Simulator
        vwap_dist = 0.0
        iv_rank = 0.5

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
        else:
            price = float(res)

        # 3. Update Rolling Windows (300-bar max)
        self.recent_prices.append(price)
        self.recent_vwap_dist.append(vwap_dist)
        self.recent_iv_rank.append(iv_rank)

        if len(self.recent_prices) > 300:
            self.recent_prices.pop(0)
            self.recent_vwap_dist.pop(0)
            self.recent_iv_rank.pop(0)

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
    init_price = float(getattr(simulator, "price", 100.0))
    prices_buffer = [init_price] * 300
    vwap_buffer = [0.0] * 300
    iv_buffer = [0.5] * 300

    # Build seed DataFrame for 5m models that require `df_5m` in __init__
    seed_df = pd.DataFrame(
        {
            "close": prices_buffer,
            "high": prices_buffer,
            "low": prices_buffer,
            "volume": [1000.0] * 300,
            "vwap_dist": vwap_buffer,
            "iv_rank": iv_buffer,
        }
    )

    def instantiate_env(env_cls):
        sig = inspect.signature(env_cls.__init__)
        params = sig.parameters

        kwargs = {}
        if "is_eval" in params:
            kwargs["is_eval"] = True
        if "train" in params:
            kwargs["train"] = False

        # If model expects `df_5m` (5m environment)
        if "df_5m" in params:
            kwargs["df_5m"] = seed_df.copy()

        # If model expects raw price lists (1d environment)
        if "spy_prices" in params:
            kwargs["spy_prices"] = prices_buffer
        elif "stock_prices" in params:
            kwargs["stock_prices"] = prices_buffer

        return env_cls(**kwargs)

    bull_env = instantiate_env(bull_env_cls)
    bear_env = instantiate_env(bear_env_cls)

    bull_vec_env = DummyVecEnv([lambda: bull_env])
    bear_vec_env = DummyVecEnv([lambda: bear_env])

    # Handle Bull Agent Normalization
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

    # Handle Bear Agent Normalization
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

    start_base_date = datetime.now().date() - timedelta(
        days=300 if interval == "1d" else 0
    )
    init_dt = datetime.combine(
        start_base_date, time(9, 30) if interval == "5m" else time(0, 0)
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
        bull_model_file="best_model_bi_v2(110).zip",
        bear_model_file="best_model_only_puts(135).zip",
    ),
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


def step_agent(
    npc: PPO, 
    norm_env: Optional[VecNormalize], 
    env: Any, 
    current_price: float, 
    agent_name: str
) -> dict:
    prev_contracts = env.contracts
    
    # Get raw observation from environment
    raw_obs = env._get_normalized_observation()
    
    # Ensure vector format for batch inference
    obs_batch = np.array([raw_obs])
    
    # Apply VecNormalize ONLY if a normalization object exists (1-day model)
    if norm_env is not None:
        norm_obs = norm_env.normalize_obs(obs_batch)
    else:
        norm_obs = obs_batch  # 5-minute model (already normalized internally)

    # Predict action
    action, _ = npc.predict(norm_obs, deterministic=True)
    act_vec = action[0] if action.ndim > 1 else action

    # Advance environment state
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