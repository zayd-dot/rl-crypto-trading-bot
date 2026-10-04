"""Custom OpenAI Gym environment for crypto futures trading."""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class CryptoTradingEnv(gym.Env):
    """
    RL environment for BTC futures trading.

    Actions:
        0 - Hold (do nothing)
        1 - Open Long position
        2 - Open Short position
        3 - Close current position

    Observation:
        Sliding window of OHLCV + technical indicators.

    Reward:
        Realized P&L on close + unrealized P&L on hold.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, df: pd.DataFrame, window_size: int = 50):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.window_size = window_size

        # Feature columns = everything except timestamp
        self.feature_cols = [c for c in df.columns if c != "timestamp"]
        n_features = len(self.feature_cols)

        # Action & observation spaces
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(window_size, n_features),
            dtype=np.float32,
        )

        # Internal state
        self.current_step = window_size
        self.current_position = None  # None, 'long', or 'short'
        self.entry_price = None

        # Tracking
        self.total_reward = 0.0
        self.trades = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.current_position = None
        self.entry_price = None
        self.total_reward = 0.0
        self.trades = []
        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        start = self.current_step - self.window_size
        end = self.current_step
        obs = self.df[self.feature_cols].iloc[start:end].values
        return obs.astype(np.float32)

    def step(self, action: int):
        reward = 0.0
        done = self.current_step >= len(self.df) - 1
        current_price = self.df["close"].values[self.current_step]
        prev_price = self.df["close"].values[self.current_step - 1]

        # --- Action logic ---
        # Open Long
        if action == 1 and self.current_position != "long":
            if self.current_position == "short":
                reward += self.entry_price - current_price  # close short first
            self.current_position = "long"
            self.entry_price = current_price
            self.trades.append(("long", self.current_step, current_price))

        # Open Short
        elif action == 2 and self.current_position != "short":
            if self.current_position == "long":
                reward += current_price - self.entry_price  # close long first
            self.current_position = "short"
            self.entry_price = current_price
            self.trades.append(("short", self.current_step, current_price))

        # Close Position
        elif action == 3 and self.current_position is not None:
            if self.current_position == "long":
                reward += current_price - self.entry_price
            elif self.current_position == "short":
                reward += self.entry_price - current_price
            self.trades.append(("close", self.current_step, current_price))
            self.current_position = None
            self.entry_price = None

        # Ongoing position reward (mark-to-market)
        if self.current_position == "long":
            reward += current_price - prev_price
        elif self.current_position == "short":
            reward += prev_price - current_price

        self.total_reward += reward
        self.current_step += 1

        truncated = False
        info = {
            "position": self.current_position,
            "total_reward": self.total_reward,
            "n_trades": len(self.trades),
        }

        obs = self._get_observation() if not done else np.zeros_like(self._get_observation())
        return obs, reward, done, truncated, info
