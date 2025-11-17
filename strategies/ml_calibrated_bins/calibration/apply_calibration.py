#!/usr/bin/env python3
"""
Apply calibrated binning to predictions.

Load monthly bin metadata and apply calibrator + bin assignment to new predictions.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .bin_manager import BinMetadata, load_bin_metadata
from .calibrators import assign_bins


def apply_calibration_and_bins(
    y_pred: Iterable,
    metadata: BinMetadata,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply calibration and bin assignment to predictions.

    Args:
        y_pred: Raw model predictions (probabilities 0-1)
        metadata: BinMetadata containing calibrator and bin edges

    Returns:
        Tuple of (calibrated_probs, bin_indices)
        - calibrated_probs: Calibrated probabilities (numpy array)
        - bin_indices: Bin assignments 1..n_bins (numpy array)
    """
    # Apply calibrator
    p_cal = metadata.calibrator.predict_proba(y_pred)

    # Assign bins
    q_bin = assign_bins(p_cal, metadata.bin_edges, metadata.n_bins)

    return p_cal, q_bin


def score_predictions_with_bins(
    df: pd.DataFrame,
    bins_root: str,
    dataset: str,
    run_name: str,
    bin_month: str,
    *,
    pred_col: str = 'y_pred',
    timestamp_col: str = 'timestamp',
) -> pd.DataFrame:
    """Score predictions DataFrame with calibrated bins.

    Args:
        df: DataFrame with predictions
        bins_root: Root directory for bin files
        dataset: Dataset identifier
        run_name: Run name
        bin_month: Month of bins to use (YYYY-MM)
        pred_col: Name of prediction column
        timestamp_col: Name of timestamp column

    Returns:
        DataFrame with added columns: p_cal, q_bin
    """
    # Load bin metadata
    metadata = load_bin_metadata(bins_root, dataset, run_name, bin_month)

    # Apply calibration
    p_cal, q_bin = apply_calibration_and_bins(df[pred_col], metadata)

    # Add to dataframe
    result = df.copy()
    result['p_cal'] = p_cal
    result['q_bin'] = q_bin

    return result


def get_current_bin_month(timestamp: pd.Timestamp | str) -> str:
    """Get the bin month string for a given timestamp.

    Args:
        timestamp: Timestamp to convert

    Returns:
        Month string in YYYY-MM format
    """
    ts = pd.to_datetime(timestamp)
    return ts.strftime('%Y-%m')


__all__ = [
    'apply_calibration_and_bins',
    'score_predictions_with_bins',
    'get_current_bin_month',
]
