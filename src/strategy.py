"""Trading strategy with technical analysis signals and risk management."""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TradingStrategy:
    """Rule-based trading signals and position sizing."""

    def __init__(self, client, config):
        self.client = client
        self.config = config
        self.balance = config.INITIAL_BALANCE

    def should_enter_long(self, df: pd.DataFrame) -> bool:
        """RSI oversold + MACD bullish crossover."""
        return (
            df["RSI"].iloc[-1] < 40
            and df["MACD"].iloc[-1] > df["MACD_Signal"].iloc[-1]
        )

    def should_enter_short(self, df: pd.DataFrame) -> bool:
        """RSI overbought + MACD bearish crossover."""
        return (
            df["RSI"].iloc[-1] > 60
            and df["MACD"].iloc[-1] < df["MACD_Signal"].iloc[-1]
        )

    def calculate_position_size(self, current_price: float) -> float:
        """Calculate position size based on risk management rules."""
        max_position = (self.balance * self.config.MAX_RISK_PER_TRADE) / current_price

        try:
            symbol_info = self.client.futures_exchange_info()
            for symbol in symbol_info["symbols"]:
                if symbol["symbol"] == self.config.SYMBOL:
                    precision = int(symbol["filters"][2]["stepSize"].find("1") - 1)
                    max_position = round(max_position, precision)
                    break
        except Exception as e:
            logger.error(f"Error fetching symbol precision: {e}")

        return min(max_position, 1.0)  # Cap at 1 BTC for safety

    def execute_trade(self, action: str, symbol: str, quantity: float,
                      current_price: float, entry_price: float = None) -> dict:
        """Execute a trade on Binance Futures."""
        try:
            if symbol == "BTCUSDT":
                quantity = round(quantity, 6)

            if quantity <= 0:
                logger.warning(f"Invalid trade quantity: {quantity}")
                return None

            # Set leverage
            self.client.futures_change_leverage(
                symbol=symbol, leverage=self.config.LEVERAGE
            )

            logger.info(f"Balance before trade: ${self.balance:.2f}")

            if action == "long":
                order = self.client.futures_create_order(
                    symbol=symbol, side="BUY", type="MARKET", quantity=quantity
                )
                self.balance -= quantity * current_price
            elif action == "short":
                order = self.client.futures_create_order(
                    symbol=symbol, side="SELL", type="MARKET", quantity=quantity
                )
                self.balance += quantity * current_price
            else:
                return None

            logger.info(f"{action.upper()} order executed: {order}")

            # Calculate P&L if closing a position
            if entry_price is not None:
                if action == "long":
                    pnl = (current_price - entry_price) * quantity
                else:
                    pnl = (entry_price - current_price) * quantity
                self.balance += pnl
                logger.info(f"Trade P&L: {'+'if pnl >= 0 else ''}{pnl:.2f}")

            logger.info(f"Balance after trade: ${self.balance:.2f}")
            return order

        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            return None
