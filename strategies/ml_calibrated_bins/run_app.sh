#!/bin/bash
# Launch the ML Calibrated Bins Strategy Streamlit app
#
# Usage:
#   ./strategies/ml_calibrated_bins/run_app.sh
#

cd "$(dirname "$0")/../.."
streamlit run strategies/ml_calibrated_bins/app_bins_strategy.py
