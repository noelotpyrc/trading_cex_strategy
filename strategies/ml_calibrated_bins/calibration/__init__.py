"""Calibration and binning utilities for trading strategies."""

from .bin_manager import (
    BinMetadata,
    save_bin_metadata,
    load_bin_metadata,
    list_available_bins,
    get_bin_path,
)
from .calibrators import (
    PlattCalibrator,
    IsotonicCalibrator,
    fit_calibrator,
    quantile_edges,
    assign_bins,
)
from .apply_calibration import (
    apply_calibration_and_bins,
    score_predictions_with_bins,
    get_current_bin_month,
)

__all__ = [
    # Bin management
    "BinMetadata",
    "save_bin_metadata",
    "load_bin_metadata",
    "list_available_bins",
    "get_bin_path",
    # Calibrators
    "PlattCalibrator",
    "IsotonicCalibrator",
    "fit_calibrator",
    "quantile_edges",
    "assign_bins",
    # Application
    "apply_calibration_and_bins",
    "score_predictions_with_bins",
    "get_current_bin_month",
]
