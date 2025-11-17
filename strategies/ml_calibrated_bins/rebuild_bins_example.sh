#!/bin/bash
# Example script for rebuilding monthly bins
#
# Usage:
#   ./rebuild_bins_example.sh 2025-06
#   ./rebuild_bins_example.sh 2025-06 2025-07 2025-08  # Multiple months
#

# Configuration
PRED_DUCKDB="/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_prediction_classifier.duckdb"
OHLCV_DUCKDB="/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_ohlcv.duckdb"
MODEL_PATH="/Volumes/Extreme SSD/trading_data/cex/models/binance_btcusdt_perp_1h_original/run_20251102_231428_lgbm_y_tp_before_sl_u0.04_d0.02_24h_binary/model.txt"
DATASET="binance_btcusdt_perp_1h"
TARGET_KEY="y_tp_before_sl_u0.04_d0.02_24h"
LOOKBACK_MONTHS=12
N_BINS=20
METHOD="platt"

# Check arguments
if [ $# -eq 0 ]; then
    echo "Usage: $0 <bin-month> [<bin-month> ...]"
    echo "Example: $0 2025-06"
    echo "Example: $0 2025-06 2025-07 2025-08"
    exit 1
fi

# Run rebuild
python -m strategies.ml_calibrated_bins.rebuild_bins \
  --pred-duckdb "$PRED_DUCKDB" \
  --ohlcv-duckdb "$OHLCV_DUCKDB" \
  --model-path "$MODEL_PATH" \
  --dataset "$DATASET" \
  --target-key "$TARGET_KEY" \
  --bin-month "$@" \
  --lookback-months $LOOKBACK_MONTHS \
  --n-bins $N_BINS \
  --method $METHOD
