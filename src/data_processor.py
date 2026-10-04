"""Data fetching and technical indicator processing."""

import logging
import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands

logger = logging.getLogger(__name__)


class DataProcessor:
    """Fetches market data and computes technical indicators."""

    @staticmethod
    def fetch_data(client, symbol: str, interval: str, limit: int = 1000) -> pd.DataFrame:
        """Fetch OHLCV data from Binance Futures API."""
        try:
            klines = client.futures_klines(symbol=symbol, interval=interval, limit=limit)

            if not klines:
                logger.warning(f"No klines data fetched for {symbol}")
                return pd.DataFrame()

            df = pd.DataFrame(klines, columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore",
            ])

            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)

            df = df[["timestamp", "open", "high", "low", "close", "volume"]]
            return df.replace(0, np.nan).dropna()

        except Exception as e:
            logger.error(f"Data fetch error for {symbol}: {e}")
            return pd.DataFrame()

    @staticmethod
    def fetch_from_csv(path: str) -> pd.DataFrame:
        """Load OHLCV data from a CSV file (for offline backtesting)."""
        df = pd.read_csv(path, parse_dates=["timestamp"])
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        return df

    @staticmethod
    def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add RSI, MACD, and Bollinger Band indicators."""
        if df.empty:
            logger.warning("Empty dataframe passed to add_technical_indicators")
            return df

        df = df.copy()

        # RSI
        df["RSI"] = RSIIndicator(df["close"]).rsi()

        # MACD
        macd = MACD(df["close"])
        df["MACD"] = macd.macd()
        df["MACD_Signal"] = macd.macd_signal()

        # Bollinger Bands
        bb = BollingerBands(df["close"])
        df["BB_High"] = bb.bollinger_hband()
        df["BB_Low"] = bb.bollinger_lband()

        return df.replace([np.inf, -np.inf], 0).fillna(0)
