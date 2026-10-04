"""Backtesting engine for evaluating trained RL agents."""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from .environment import CryptoTradingEnv

logger = logging.getLogger(__name__)


class Backtester:
    """Run a trained PPO agent on historical data and report performance."""

    def __init__(self, model_path: str, config):
        self.model = PPO.load(model_path)
        self.config = config

    def run(self, df: pd.DataFrame) -> dict:
        """Backtest the agent on a DataFrame of OHLCV + indicators."""
        env = CryptoTradingEnv(df, self.config.WINDOW_SIZE)
        obs, _ = env.reset()

        actions_taken = []
        rewards = []
        prices = []
        positions = []

        done = False
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            action = int(action)
            obs, reward, done, truncated, info = env.step(action)

            actions_taken.append(action)
            rewards.append(reward)
            prices.append(df["close"].values[env.current_step - 1])
            positions.append(info["position"])

        total_reward = sum(rewards)
        n_trades = info["n_trades"]
        sharpe = self._sharpe_ratio(rewards)

        results = {
            "total_reward": total_reward,
            "n_trades": n_trades,
            "sharpe_ratio": sharpe,
            "actions": actions_taken,
            "rewards": rewards,
            "prices": prices,
            "positions": positions,
        }

        logger.info(f"Backtest complete: reward={total_reward:.2f}, "
                     f"trades={n_trades}, sharpe={sharpe:.3f}")
        return results

    @staticmethod
    def _sharpe_ratio(rewards, risk_free_rate=0.0):
        """Annualized Sharpe ratio (assumes minute-level data)."""
        returns = np.array(rewards)
        if returns.std() == 0:
            return 0.0
        excess = returns.mean() - risk_free_rate
        return (excess / returns.std()) * np.sqrt(525600)  # minutes in a year

    @staticmethod
    def plot_results(results: dict, save_path: str = None):
        """Plot backtest equity curve and trade markers."""
        prices = results["prices"]
        actions = results["actions"]
        cumulative = np.cumsum(results["rewards"])

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        # Price + trade markers
        ax1.plot(prices, color="steelblue", linewidth=0.8, label="Price")
        longs = [i for i, a in enumerate(actions) if a == 1]
        shorts = [i for i, a in enumerate(actions) if a == 2]
        closes = [i for i, a in enumerate(actions) if a == 3]
        ax1.scatter(longs, [prices[i] for i in longs], marker="^",
                    color="green", s=30, label="Long", zorder=5)
        ax1.scatter(shorts, [prices[i] for i in shorts], marker="v",
                    color="red", s=30, label="Short", zorder=5)
        ax1.scatter(closes, [prices[i] for i in closes], marker="x",
                    color="orange", s=30, label="Close", zorder=5)
        ax1.set_title("Price & Trade Signals")
        ax1.legend()

        # Cumulative reward
        ax2.plot(cumulative, color="darkgreen", linewidth=1)
        ax2.axhline(0, color="gray", linestyle="--", linewidth=0.5)
        ax2.set_title("Cumulative Reward")
        ax2.set_xlabel("Step")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
