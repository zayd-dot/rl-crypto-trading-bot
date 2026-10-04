"""Configuration for RL Crypto Trading Bot."""

from dataclasses import dataclass
import os


@dataclass
class TradingConfig:
    """Trading parameters and API configuration."""

    # Trading pair
    SYMBOL: str = "BTCUSDT"
    INTERVAL: str = "1m"

    # RL Environment
    WINDOW_SIZE: int = 50
    INITIAL_BALANCE: float = 1_000_000

    # Risk Management
    MAX_RISK_PER_TRADE: float = 0.4  # % of total balance
    LEVERAGE: int = 20

    # PPO Hyperparameters
    LEARNING_RATE: float = 3e-4
    N_STEPS: int = 128
    BATCH_SIZE: int = 64
    N_EPOCHS: int = 10
    CLIP_RANGE: float = 0.2
    TOTAL_TIMESTEPS: int = 10_000

    # Paths
    CHECKPOINT_DIR: str = "checkpoints"
    MODEL_PATH: str = "checkpoints/trading_model"
    LOG_FILE: str = "logs/trading_bot.log"

    # Binance API (load from environment variables for security)
    BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY", "")
    BINANCE_API_SECRET: str = os.getenv("BINANCE_API_SECRET", "")
    USE_TESTNET: bool = True


@dataclass
class BacktestConfig:
    """Backtesting parameters."""

    INITIAL_CAPITAL: float = 10_000
    COMMISSION: float = 0.001  # 0.1% per trade
    SLIPPAGE: float = 0.0005  # 0.05%
