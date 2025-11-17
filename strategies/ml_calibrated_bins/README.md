# ML Calibrated Bins

Monthly rolling calibration and quantile binning for machine learning predictions.

## Overview

This module provides tools to:
1. **Calibrate predictions** - Transforms raw ML probabilities to well-calibrated probabilities using Platt scaling or Isotonic regression
2. **Create quantile bins** - Divides calibrated probabilities into equal-count bins (default: 20)
3. **Persist monthly bins** - Saves calibration metadata for production use

The bins are rebuilt monthly using a rolling 12-month calibration window.

## Quick Start

Use the provided example script:

```bash
# Edit rebuild_bins_example.sh to configure your paths
# Then run for single or multiple months:
./strategies/ml_calibrated_bins/rebuild_bins_example.sh 2025-06
./strategies/ml_calibrated_bins/rebuild_bins_example.sh 2025-06 2025-07 2025-08
```

Or run the command directly (see `rebuild_bins_example.sh` for the full command template).

## Workflow

### Step 1: Rebuild Monthly Bins

Rebuild calibrated bins from predictions and targets:

```bash
python -m strategies.ml_calibrated_bins.rebuild_bins \
  --pred-duckdb "/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_prediction_classifier.duckdb" \
  --ohlcv-duckdb "/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_ohlcv.duckdb" \
  --model-path "/Volumes/Extreme SSD/trading_data/cex/models/binance_btcusdt_perp_1h_original/run_20251102_231428_lgbm_y_tp_before_sl_u0.04_d0.02_24h_binary/model.txt" \
  --dataset binance_btcusdt_perp_1h \
  --target-key y_tp_before_sl_u0.04_d0.02_24h \
  --bin-month 2025-11 \
  --lookback-months 12 \
  --n-bins 20 \
  --method platt
```

**Backfill multiple months:**
```bash
python -m strategies.ml_calibrated_bins.rebuild_bins \
  --pred-duckdb "..." \
  --ohlcv-duckdb "..." \
  --model-path "..." \
  --dataset binance_btcusdt_perp_1h \
  --target-key y_tp_before_sl_u0.04_d0.02_24h \
  --bin-month 2025-05 2025-06 2025-07 2025-08 2025-09 2025-10 2025-11
```

**What this does:**
1. Loads predictions from DuckDB (falls back to pred_train.csv if needed)
2. Loads actuals (y_true) from targets table
3. Fits Platt/Isotonic calibrator on (y_pred, y_true) pairs
4. Derives quantile bin edges from calibrated probabilities
5. Saves to `bins/{dataset}/{run_name}/{YYYY-MM}.pkl`

**Note on predictions:** The script automatically falls back to `pred_train.csv` in the model directory if predictions aren't available in DuckDB for the full calibration window.

### Step 2: Inspect Bins

View bin metadata and test calibration:

```bash
# Basic inspection
python strategies/ml_calibrated_bins/inspect_bins.py 2025-05

# With calibrator test
python strategies/ml_calibrated_bins/inspect_bins.py 2025-05 --test
```

### Step 3: Analyze Strategy with Streamlit App

Launch the interactive Streamlit app to visualize and analyze your strategy:

```bash
./strategies/ml_calibrated_bins/run_app.sh
# Or directly:
streamlit run strategies/ml_calibrated_bins/app_bins_strategy.py
```

**Features:**
- Configure which bins trigger long/short entries
- Visualize entry signals on price charts
- Analyze bin distributions and calibrated probabilities
- Export signals to CSV
- Interactive date range selection

### Step 4: Apply Bins to New Predictions (Programmatic)

Use the calibrated bins in your code:

```python
from strategies.ml_calibrated_bins.calibration import (
    load_bin_metadata,
    apply_calibration_and_bins,
)

# Load bin metadata
bin_meta = load_bin_metadata(
    bins_root='./strategies/ml_calibrated_bins/bins',
    dataset='binance_btcusdt_perp_1h',
    run_name='run_20251102_231428_lgbm_...',
    bin_month='2025-05'
)

# Apply to predictions
p_cal, q_bin = apply_calibration_and_bins(y_pred, bin_meta)

# Result:
# p_cal: Calibrated probabilities (0-1)
# q_bin: Bin assignments (1-20)
```

Or use the DataFrame helper:

```python
from strategies.ml_calibrated_bins.calibration import score_predictions_with_bins
import pandas as pd

# Load your predictions
df = pd.read_csv('predictions.csv')  # Must have 'timestamp' and 'y_pred'

# Apply calibration and binning
df_scored = score_predictions_with_bins(
    df,
    bins_root='./strategies/ml_calibrated_bins/bins',
    dataset='binance_btcusdt_perp_1h',
    run_name='run_20251102_231428_lgbm_...',
    bin_month='2025-05',
)

# Result: df with added columns 'p_cal' and 'q_bin'
```

## Directory Structure

```
strategies/ml_calibrated_bins/
├── calibration/
│   ├── bin_manager.py          # Save/load bin metadata
│   ├── calibrators.py          # Platt/Isotonic calibrators
│   └── apply_calibration.py    # Apply bins to predictions
├── bins/                        # Persisted bin metadata
│   └── {dataset}/{run_name}/{YYYY-MM}.pkl
├── rebuild_bins.py              # Monthly bin rebuild script
├── rebuild_bins_example.sh      # Example rebuild script
├── inspect_bins.py              # Bin inspection tool
├── app_bins_strategy.py         # Streamlit app for strategy analysis
├── run_app.sh                   # Launch Streamlit app
└── README.md                    # This file
```

## Bin Metadata Format

Each `.pkl` file contains a dictionary with:

```python
{
    'bin_month': '2025-05',
    'n_bins': 20,
    'calibration_method': 'platt',
    'bin_edges': np.array([...]),           # shape (19,) for 20 bins
    'calibrator': PlattCalibrator(...),     # Fitted calibrator object
    'cal_window_start': '2024-05-01',
    'cal_window_end': '2025-04-30 23:00:00',
    'cal_rows': 8758,
    'created_at': '2025-11-16 11:30:00',
    'model_path': '/Volumes/.../model.txt',
    'dataset': 'binance_btcusdt_perp_1h',
    'run_name': 'run_20251102_231428_...',
}
```

## Calibration Methods

### Platt Scaling (Default)
- **Method**: Logistic regression on logit-transformed predictions
- **Pros**: Fast, parametric, stable extrapolation
- **Use when**: General production use, default choice

### Isotonic Regression
- **Method**: Non-parametric monotonic calibration
- **Pros**: Flexible, can capture complex relationships
- **Cons**: Can overfit on small datasets
- **Use when**: Large calibration windows (>10k samples)

## Data Requirements

### Targets Table

The rebuild script expects targets to be in the `targets` table (in OHLCV DuckDB):

```sql
CREATE TABLE targets (
    timestamp TIMESTAMP NOT NULL,
    dataset TEXT NOT NULL,
    target_key TEXT NOT NULL,
    target_value DOUBLE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (timestamp, dataset, target_key)
);
```

Use `backfill_targets.py` from the `trading_cex_data_processing` repo to populate this table.

See: `trading_cex_data_processing/docs/targets_table.md`

### Predictions Sources

The rebuild script loads predictions from:
1. **DuckDB predictions table** (primary)
2. **pred_train.csv** (fallback if DuckDB has gaps)

The script automatically merges both sources, preferring DuckDB data for duplicates.

## Monthly Automation

Set up a cron job to rebuild bins automatically:

```bash
# On the 1st of each month at midnight
0 0 1 * * cd /path/to/trading_cex_strategy && \
  ./venv/bin/python -m strategies.ml_calibrated_bins.rebuild_bins \
  --pred-duckdb "..." \
  --ohlcv-duckdb "..." \
  --model-path "..." \
  --dataset binance_btcusdt_perp_1h \
  --target-key y_tp_before_sl_u0.04_d0.02_24h \
  --bin-month $(date -d "last month" +%Y-%m) \
  --lookback-months 12 \
  --n-bins 20 \
  --method platt
```

## Troubleshooting

**FileNotFoundError: Bin metadata not found**

Make sure you've run `rebuild_bins.py` for the month you're trying to use. Check available bins:

```python
from strategies.ml_calibrated_bins.calibration import list_available_bins

bins = list_available_bins('./strategies/ml_calibrated_bins/bins')
print(bins)
```

**ValueError: No predictions/targets found**

Verify your DuckDB tables have data for the requested time range:

```sql
-- Check predictions
SELECT MIN(ts), MAX(ts), COUNT(*) FROM predictions WHERE model_path = '...';

-- Check targets
SELECT MIN(timestamp), MAX(timestamp), COUNT(*)
FROM targets
WHERE dataset = 'binance_btcusdt_perp_1h'
  AND target_key = 'y_tp_before_sl_u0.04_d0.02_24h';
```

**Only getting partial calibration window**

If the predictions table doesn't have historical data, the script will automatically fall back to `pred_train.csv` in the model directory. Check the output for messages like:

```
Loaded 720 predictions from DuckDB, added 8038 from pred_train.csv
```

## Performance

- **Bin rebuild**: ~1-5s per month (depends on calibration window size)
- **Bin application**: ~100-500ms per 1000 predictions
- **Bin file size**: ~50-200 KB per month
