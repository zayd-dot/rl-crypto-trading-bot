# RL Crypto Trading Bot

A reinforcement-learning project using PPO and a custom Gymnasium environment for BTC market decision-making. The project includes historical-data loading, technical indicators, four discrete actions, model training, and a backtesting interface.

## Pipeline

1. Load OHLCV data from CSV or Binance.
2. Calculate RSI, MACD, MACD signal, and Bollinger Bands.
3. Standardize selected features.
4. Construct 50-step observation windows.
5. Train a PPO policy to choose hold, long, short, or close.
6. Run the policy through historical data and record actions and cumulative simulation reward.

## Architecture

| Component | Implementation |
|---|---|
| Environment | Custom Gymnasium environment |
| Observation | 50-step market-feature window |
| Actions | Hold, long, short, close |
| Algorithm | PPO from Stable-Baselines3 |
| Policy | `MlpPolicy` |
| Learning rate | 0.0003 |
| Rollout steps | 128 |
| Batch size | 64 |
| PPO clipping | 0.2 |

The environment tracks position transitions and assigns a simulation reward from price movements and position changes. Reward values are simulator outputs, not currency-denominated portfolio returns.

## Reported Experiment

The project reports positive cumulative simulation reward on a six-month BTC hourly-data backtest.

| Item | Reported value |
|---|---:|
| Cumulative simulation reward | **18.62** |
| Recorded trading events | **1,676** |
| Hourly candles | **4,182** |
| Training timesteps | **50,000** |

## Quick Start

Provide a CSV containing `timestamp`, `open`, `high`, `low`, `close`, and `volume`.

```bash
python -m pip install -r requirements.txt
python train.py --csv data/btcusdt_1h.csv --timesteps 50000
```

Run the historical backtesting interface:

```bash
mkdir -p results
python backtest.py --csv data/btcusdt_1h.csv --model checkpoints/trading_model
```

## Outputs

- Saved PPO model.
- Action sequence and position tracking.
- Cumulative simulation reward.
- Price/action visualization and reward curve.

The repository also contains Binance Futures testnet integration code and configurable position-sizing and leverage settings.

## Project Structure

| File | Purpose |
|---|---|
| `src/data_processor.py` | Data loading and technical indicators |
| `src/environment.py` | Observation, actions, transitions, and reward |
| `train.py` | PPO training |
| `src/backtest.py` | Historical policy evaluation and plots |
| `backtest.py` | Backtesting CLI |
| `src/strategy.py` | Exchange integration and sizing functions |
| `live_trade.py` | Testnet integration entry point |
| `configs/config.py` | Trading and PPO settings |

## Technologies

Python, Gymnasium, Stable-Baselines3, PyTorch, Pandas, NumPy, Scikit-learn, technical-analysis utilities, and Binance API integration.
