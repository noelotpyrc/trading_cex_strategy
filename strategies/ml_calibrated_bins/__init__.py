"""
ML Calibrated Bins

Monthly rolling calibration and quantile binning for machine learning predictions:
- Calibration: Platt scaling or Isotonic regression
- Binning: Quantile-based bins (default: 20)
- Monthly updates: Rolling 12-month calibration window

Provides tools to transform raw ML predictions into calibrated probabilities
and assign them to quantile bins for downstream strategy use.
"""

from .calibration import (
    BinMetadata,
    save_bin_metadata,
    load_bin_metadata,
    list_available_bins,
    PlattCalibrator,
    IsotonicCalibrator,
    fit_calibrator,
    apply_calibration_and_bins,
    score_predictions_with_bins,
)

__all__ = [
    # Bin management
    "BinMetadata",
    "save_bin_metadata",
    "load_bin_metadata",
    "list_available_bins",
    # Calibrators
    "PlattCalibrator",
    "IsotonicCalibrator",
    "fit_calibrator",
    # Application
    "apply_calibration_and_bins",
    "score_predictions_with_bins",
]
