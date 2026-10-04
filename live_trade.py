"""Live trading bot — connects to Binance Futures testnet and trades with a trained PPO model."""

import argparse
import logging
import os
import time

import numpy as np
from binance.client import Client
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import PPO

from configs.config import TradingConfig
from src.data_processor import DataProcessor
from src.strategy import TradingStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/trading_bot.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

ACTION_MAP = {0: "hold", 1: "long", 2: "short", 3: "close"}


class LiveTrader:
    """Runs a trained PPO agent in a live trading loop."""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.scaler = StandardScaler()

        # Binance client
        self.client = Client(
            config.BINANCE_API_KEY,
            config.BINANCE_API_SECRET,
            testnet=config.USE_TESTNET,
        )
        if config.USE_TESTNET:
            self.client.FUTURES_URL = "https://testnet.binancefuture.com"

        self.strategy = TradingStrategy(self.client, config)

        # Load trained model
        if not os.path.exists(f"{config.MODEL_PATH}.zip"):
            raise FileNotFoundError(
                f"No trained model at {config.MODEL_PATH}.zip — run train.py first"
            )
        self.model = PPO.load(config.MODEL_PATH)
        logger.info("Loaded trained PPO model")

    def run(self, max_iterations: int = None):
        """Main trading loop."""
        iteration = 0
        retry_count = 0
        max_retries = 5

        while max_iterations is None or iteration < max_iterations:
            try:
                # Fetch latest data
                df = DataProcessor.fetch_data(
                    self.client, self.config.SYMBOL, self.config.INTERVAL
                )

                if df.empty:
                    retry_count += 1
                    logger.warning(f"No data fetched (retry {retry_count}/{max_retries})")
                    if retry_count > max_retries:
                        logger.error("Max retries reached. Stopping.")
                        break
                    time.sleep(60)
                    continue

                df = DataProcessor.add_technical_indicators(df)

                if len(df) < self.config.WINDOW_SIZE:
                    logger.warning(f"Insufficient data: {len(df)} < {self.config.WINDOW_SIZE}")
                    time.sleep(60)
                    continue

                # Normalize
                feature_cols = ["close", "RSI", "MACD", "MACD_Signal"]
                df[feature_cols] = self.scaler.fit_transform(df[feature_cols])

                current_price = df["close"].iloc[-1]

                # Build observation
                obs = (
                    df.tail(self.config.WINDOW_SIZE)
                    .drop(columns=["timestamp"])
                    .values.astype(np.float32)
                )

                # Predict action
                action, _ = self.model.predict(obs, deterministic=True)
                action = int(action)
                action_str = ACTION_MAP[action]

                logger.info(f"Step {iteration} | Action: {action_str} | Price: {current_price:.2f}")

                # Execute trade
                position_size = self.strategy.calculate_position_size(current_price)

                if action_str in ("long", "short"):
                    self.strategy.execute_trade(
                        action_str, self.config.SYMBOL, position_size, current_price
                    )

                retry_count = 0
                iteration += 1
                time.sleep(60)

            except KeyboardInterrupt:
                logger.info("Trading stopped by user")
                break
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                retry_count += 1
                if retry_count > max_retries:
                    logger.critical("Max retries reached. Exiting.")
                    break
                time.sleep(60)


def main():
    parser = argparse.ArgumentParser(description="Run live trading bot")
    parser.add_argument("--iterations", type=int, default=None,
                        help="Max trading iterations (None = run forever)")
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    config = TradingConfig()
    trader = LiveTrader(config)
    trader.run(max_iterations=args.iterations)


if __name__ == "__main__":
    main()
