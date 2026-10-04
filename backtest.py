"""Backtest a trained PPO agent on historical data."""

import argparse
import logging

from sklearn.preprocessing import StandardScaler

from configs.config import TradingConfig, BacktestConfig
from src.data_processor import DataProcessor
from src.backtest import Backtester

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Backtest trained RL agent")
    parser.add_argument("--csv", type=str, required=True, help="Path to OHLCV CSV")
    parser.add_argument("--model", type=str, default="checkpoints/trading_model",
                        help="Path to saved PPO model")
    parser.add_argument("--save-plot", type=str, default="results/backtest.png",
                        help="Save plot to path")
    args = parser.parse_args()

    config = TradingConfig()

    # Load and process data
    df = DataProcessor.fetch_from_csv(args.csv)
    df = DataProcessor.add_technical_indicators(df)

    feature_cols = ["close", "RSI", "MACD", "MACD_Signal"]
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    logger.info(f"Backtesting on {len(df)} rows...")

    # Run backtest
    bt = Backtester(args.model, config)
    results = bt.run(df)

    print(f"\n{'='*40}")
    print(f"Total Reward:  {results['total_reward']:.2f}")
    print(f"Trades:        {results['n_trades']}")
    print(f"Sharpe Ratio:  {results['sharpe_ratio']:.3f}")
    print(f"{'='*40}\n")

    Backtester.plot_results(results, save_path=args.save_plot)


if __name__ == "__main__":
    main()
