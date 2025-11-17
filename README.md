# Trading CEX Strategy

Production-ready trading strategy framework for cryptocurrency futures.

## Overview

This repository provides a **strategy-based architecture** for converting model predictions into trading signals. Each strategy is self-contained with its own logic, configuration, and artifacts.

**Current Strategies:**
1. **ML Calibrated Bins** - Uses machine learning predictions with monthly calibration and quantile binning

**Future Strategies:**
- Technical indicators (RSI, MACD, Bollinger Bands)
- Mean reversion
- Arbitrage
- Multi-model ensembles

### Philosophy

- **Strategy-agnostic architecture** - Not all strategies use ML or calibration
- **Self-contained strategies** - Each strategy owns its logic and artifacts
- **Shared utilities** - Common code lives in `common/`
- **Modular design** - Easy to add new strategies without refactoring

## Directory Structure

```
trading_cex_strategy/
├── strategies/                        # All trading strategies
│   └── ml_calibrated_bins/           # Strategy 1: ML with calibrated bins
│       ├── calibration/              # Calibrator logic
│       │   ├── bin_manager.py       # Save/load bin metadata
│       │   ├── calibrators.py       # Platt/Isotonic calibrators
│       │   └── apply_calibration.py # Apply bins to predictions
│       ├── bins/                     # Persisted bin metadata
│       │   └── {dataset}/{run_name}/{YYYY-MM}.pkl
│       ├── signals.py                # Signal generation
│       └── rebuild_bins.py           # Monthly bin rebuild script
├── common/                            # Shared utilities
│   └── base.py                       # Base classes (SignalType, etc.)
├── tests/                             # Unit tests
└── utils/                             # General utilities
```

## Installation

```bash
cd /Users/noel/projects/trading_cex_strategy
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Strategy 1: ML Calibrated Bins

Binary trading strategy using ML predictions with monthly rolling calibration.

### How It Works

```
┌──────────────────────────────────┐
│  trading_cex_inference           │
│  → Raw predictions (y_pred)      │
└──────────────────────────────────┘
            ↓
┌──────────────────────────────────┐
│  Monthly Bin Rebuild             │
│  1. Fetch 12-month history       │
│  2. Fit Platt/Isotonic calibrator│
│  3. Derive quantile bin edges    │
│  4. Save bins/*.pkl              │
└──────────────────────────────────┘
            ↓
┌──────────────────────────────────┐
│  Signal Generation               │
│  1. Load monthly bins            │
│  2. Apply calibration            │
│  3. Assign bins (1-20)           │
│  4. Generate signals:            │
│     - Bin 19+ → Long             │
│     - Bin 1-2 → Short            │
└──────────────────────────────────┘
```

### Usage

#### 1. Rebuild Bins Monthly

Rebuild bins using 12-month rolling calibration window:

```bash
python strategies/ml_calibrated_bins/rebuild_bins.py \
  --pred-duckdb "/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_prediction_classifier.duckdb" \
  --ohlcv-duckdb "/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_ohlcv.duckdb" \
  --model-path "/Volumes/Extreme SSD/trading_data/cex/models/binance_btcusdt_perp_1h_original/run_20251102_231428_lgbm_y_tp_before_sl_u0.04_d0.02_24h_binary/model.txt" \
  --dataset binance_btcusdt_perp_1h \
  --bin-month 2025-11 \
  --lookback-months 12 \
  --n-bins 20 \
  --method platt
```

**Backfill multiple months:**

```bash
python strategies/ml_calibrated_bins/rebuild_bins.py \
  --pred-duckdb "..." \
  --ohlcv-duckdb "..." \
  --model-path "..." \
  --dataset binance_btcusdt_perp_1h \
  --bin-month 2025-05 2025-06 2025-07 2025-08 2025-09 2025-10 2025-11
```

#### 2. Apply Bins to Predictions (Python API)

```python
from strategies.ml_calibrated_bins.calibration import score_predictions_with_bins
import pandas as pd

# Load predictions
df = pd.read_csv('predictions.csv')  # Must have 'timestamp' and 'y_pred'

# Score with calibrated bins
df_scored = score_predictions_with_bins(
    df,
    bins_root='./strategies/ml_calibrated_bins/bins',
    dataset='binance_btcusdt_perp_1h',
    run_name='run_20251102_231428_lgbm_y_tp_before_sl_u0.04_d0.02_24h_binary',
    bin_month='2025-11',
)

# Result has columns: timestamp, y_pred, p_cal, q_bin
print(df_scored.head())
```

#### 3. Generate Trading Signals

```python
from strategies.ml_calibrated_bins.signals import BinaryBinSignal, generate_binary_bin_signals

# Configure signal rules
signal_config = BinaryBinSignal(
    long_bin_threshold=19,   # Bin >= 19 → Long
    short_bin_threshold=2,   # Bin <= 2 → Short
)

# Generate signals
df_signals = generate_binary_bin_signals(df_scored, signal_config)

# Filter for long signals
longs = df_signals[df_signals['signal'] == 1]
print(f"Long signals: {len(longs)}")

# Filter for short signals
shorts = df_signals[df_signals['signal'] == -1]
print(f"Short signals: {len(shorts)}")
```

#### 4. End-to-End Signal Generation

```python
from strategies.ml_calibrated_bins.signals import generate_signals_from_predictions, BinaryBinSignal

df_signals = generate_signals_from_predictions(
    pred_duckdb="/Volumes/.../predictions.duckdb",
    model_path="/Volumes/.../model.txt",
    bins_root='./strategies/ml_calibrated_bins/bins',
    dataset='binance_btcusdt_perp_1h',
    run_name='run_20251102_231428_lgbm_...',
    start='2025-11-01',
    end='2025-11-30',
    signal_config=BinaryBinSignal(long_bin_threshold=19, short_bin_threshold=1),
    auto_select_month=True,  # Automatically select bins by month
)
```

### Bin Metadata Format

Each monthly bin file contains:

```python
{
    'bin_month': '2025-11',
    'n_bins': 20,
    'calibration_method': 'platt',
    'bin_edges': np.array([...]),           # shape (19,) for 20 bins
    'calibrator': PlattCalibrator(...),     # Fitted calibrator object
    'cal_window_start': '2024-11-01',
    'cal_window_end': '2025-10-31 23:00:00',
    'cal_rows': 8758,
    'created_at': '2025-11-16 11:30:00',
    'model_path': '/Volumes/.../model.txt',
}
```

### Calibration Methods

**Platt Scaling (recommended):**
- Logistic regression on logit-transformed predictions
- Fast, parametric, stable extrapolation
- Default choice for production

**Isotonic Regression:**
- Non-parametric monotonic calibration
- More flexible but can overfit
- Use for larger calibration windows (>10k samples)

### Monthly Automation

```bash
# Cron job: On the 1st of each month, rebuild bins for previous month
0 0 1 * * cd /path/to/trading_cex_strategy && \
  ./venv/bin/python strategies/ml_calibrated_bins/rebuild_bins.py \
  --pred-duckdb "..." \
  --ohlcv-duckdb "..." \
  --model-path "..." \
  --dataset binance_btcusdt_perp_1h \
  --bin-month $(date -d "last month" +%Y-%m)
```

---

## Adding New Strategies

To add a new strategy (e.g., `technical_indicators`):

1. **Create strategy directory:**
   ```bash
   mkdir -p strategies/technical_indicators
   ```

2. **Implement your strategy:**
   ```python
   # strategies/technical_indicators/signals.py
   from common.base import SignalType

   def generate_rsi_signals(df, period=14, oversold=30, overbought=70):
       # Your RSI logic here
       df['signal'] = SignalType.NO_SIGNAL
       df.loc[df['rsi'] < oversold, 'signal'] = SignalType.LONG
       df.loc[df['rsi'] > overbought, 'signal'] = SignalType.SHORT
       return df
   ```

3. **Add tests and documentation**

No need to refactor existing code - each strategy is independent!

---

## Integration with Ecosystem

### From `trading_cex_inference`:
```python
# After running inference, predictions are in DuckDB
# Use ml_calibrated_bins strategy to generate signals
```

### From `trading_cex` (research):
```python
# Research workflows can import strategies
from strategies.ml_calibrated_bins import load_bin_metadata, generate_binary_bin_signals

# Load bins for analysis
metadata = load_bin_metadata(
    bins_root='./strategies/ml_calibrated_bins/bins',
    dataset='binance_btcusdt_perp_1h',
    run_name='run_20251102_...',
    bin_month='2025-11'
)
```

---

## Testing

```bash
# Run all tests
pytest tests/

# Test specific strategy
pytest tests/test_ml_calibrated_bins.py
```

---

## Version Control

**Bin Files:**
- **Option 1:** Track in git (lightweight, ~50-200 KB per month)
- **Option 2:** `.gitignore` bins/ and store separately

Current recommendation: **Track bins in git** for reproducibility.

---

## Roadmap

### Strategies
- [ ] Technical indicators strategy
- [ ] Mean reversion strategy
- [ ] Multi-model ensemble strategy
- [ ] Arbitrage strategy

### Infrastructure
- [ ] Position sizing framework
- [ ] Risk management module
- [ ] Backtest framework
- [ ] Real-time signal streaming
- [ ] Strategy performance monitoring

---

## Dependencies

- **scikit-learn** - Calibration (Platt, Isotonic)
- **pandas, numpy** - Data processing
- **duckdb** - Data loading
- **pickle** - Persistence

---

## License

MIT
