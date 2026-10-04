"""Visualization utilities for market data and trading results."""

import matplotlib.pyplot as plt
import pandas as pd


def plot_market_data(df: pd.DataFrame, save_path: str = None):
    """Plot price chart with Bollinger Bands, RSI, and MACD."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

    # Price + Bollinger Bands
    ax1.plot(df["timestamp"], df["close"], label="Close", color="steelblue")
    ax1.plot(df["timestamp"], df["BB_High"], label="BB Upper",
             color="red", linestyle="--", alpha=0.6)
    ax1.plot(df["timestamp"], df["BB_Low"], label="BB Lower",
             color="green", linestyle="--", alpha=0.6)
    ax1.fill_between(df["timestamp"], df["BB_High"], df["BB_Low"],
                      alpha=0.05, color="gray")
    ax1.set_title("Price with Bollinger Bands")
    ax1.legend()

    # RSI
    ax2.plot(df["timestamp"], df["RSI"], label="RSI", color="purple")
    ax2.axhline(y=70, color="r", linestyle="--", alpha=0.5, label="Overbought (70)")
    ax2.axhline(y=30, color="g", linestyle="--", alpha=0.5, label="Oversold (30)")
    ax2.fill_between(df["timestamp"], 30, 70, alpha=0.05, color="gray")
    ax2.set_title("RSI")
    ax2.legend()

    # MACD
    ax3.plot(df["timestamp"], df["MACD"], label="MACD", color="blue")
    ax3.plot(df["timestamp"], df["MACD_Signal"], label="Signal", color="red")
    histogram = df["MACD"] - df["MACD_Signal"]
    colors = ["green" if v >= 0 else "red" for v in histogram]
    ax3.bar(df["timestamp"], histogram, color=colors, alpha=0.3, width=0.01)
    ax3.set_title("MACD")
    ax3.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
