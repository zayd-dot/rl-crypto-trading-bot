"""Train a PPO agent for crypto futures trading."""

import argparse
import logging
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

from configs.config import TradingConfig
from src.data_processor import DataProcessor
from src.environment import CryptoTradingEnv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def train_offline(csv_path: str, config: TradingConfig):
    """Train on a local CSV file (no Binance API needed)."""
    logger.info(f"Loading data from {csv_path}")
    df = DataProcessor.fetch_from_csv(csv_path)
    df = DataProcessor.add_technical_indicators(df)

    # Normalize features
    feature_cols = ["close", "RSI", "MACD", "MACD_Signal"]
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    return _train(df, config)


def train_live(config: TradingConfig):
    """Fetch data from Binance and train."""
    from binance.client import Client

    client = Client(config.BINANCE_API_KEY, config.BINANCE_API_SECRET, testnet=config.USE_TESTNET)
    if config.USE_TESTNET:
        client.FUTURES_URL = "https://testnet.binancefuture.com"

    logger.info(f"Fetching {config.SYMBOL} data from Binance...")
    df = DataProcessor.fetch_data(client, config.SYMBOL, config.INTERVAL)

    if df.empty or len(df) < config.WINDOW_SIZE:
        logger.error(f"Insufficient data: got {len(df)} rows, need {config.WINDOW_SIZE}")
        return None

    df = DataProcessor.add_technical_indicators(df)

    feature_cols = ["close", "RSI", "MACD", "MACD_Signal"]
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    return _train(df, config)


def _train(df: pd.DataFrame, config: TradingConfig):
    """Core training loop."""
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    logger.info(f"Dataset: {len(df)} rows, {len(df.columns)} features")

    # Create environments
    env = CryptoTradingEnv(df, config.WINDOW_SIZE)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=config.LEARNING_RATE,
        n_steps=config.N_STEPS,
        batch_size=config.BATCH_SIZE,
        n_epochs=config.N_EPOCHS,
        clip_range=config.CLIP_RANGE,
        device="auto",
    )

    logger.info(f"Training PPO for {config.TOTAL_TIMESTEPS} timesteps...")
    model.learn(total_timesteps=config.TOTAL_TIMESTEPS)
    model.save(config.MODEL_PATH)
    logger.info(f"Model saved to {config.MODEL_PATH}")

    # Quick evaluation
    obs, _ = env.reset()
    total_reward = 0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(int(action))
        total_reward += reward

    logger.info(f"Evaluation — Total reward: {total_reward:.2f}, Trades: {info['n_trades']}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train RL Crypto Trading Bot")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to OHLCV CSV file for offline training")
    parser.add_argument("--timesteps", type=int, default=10_000,
                        help="Total training timesteps")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    args = parser.parse_args()

    config = TradingConfig()
    config.TOTAL_TIMESTEPS = args.timesteps
    config.LEARNING_RATE = args.lr

    if args.csv:
        train_offline(args.csv, config)
    else:
        train_live(config)


if __name__ == "__main__":
    main()
