#!/usr/bin/env python3
"""
Bin metadata management for monthly calibrated bins.

Handles persistence and loading of monthly bin metadata including:
- Calibrator objects (Platt/Isotonic)
- Bin edges (quantiles from calibrated scores)
- Calibration window metadata

File structure:
    bins/{dataset}/{run_name}/{YYYY-MM}.pkl
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


@dataclass
class BinMetadata:
    """Monthly bin metadata containing calibrator and bin edges.

    Attributes:
        bin_month: Month string (e.g., '2025-05')
        n_bins: Number of bins (e.g., 20)
        calibration_method: 'platt' or 'isotonic'
        bin_edges: Numpy array of bin edges (length n_bins-1)
        calibrator: Fitted calibrator object (PlattCalibrator or IsotonicCalibrator)
        cal_window_start: Start of calibration window
        cal_window_end: End of calibration window
        cal_rows: Number of rows in calibration window
        created_at: Timestamp when bins were created
        model_path: Full path to model file
        dataset: Dataset identifier (e.g., 'binance_btcusdt_perp_1h')
        run_name: Run name (e.g., 'run_20251102_231428_lgbm_...')
    """
    bin_month: str
    n_bins: int
    calibration_method: str
    bin_edges: np.ndarray
    calibrator: Any  # PlattCalibrator or IsotonicCalibrator
    cal_window_start: str | pd.Timestamp
    cal_window_end: str | pd.Timestamp
    cal_rows: int
    created_at: str | pd.Timestamp
    model_path: str
    dataset: str
    run_name: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'bin_month': self.bin_month,
            'n_bins': self.n_bins,
            'calibration_method': self.calibration_method,
            'bin_edges': self.bin_edges,
            'calibrator': self.calibrator,
            'cal_window_start': str(self.cal_window_start),
            'cal_window_end': str(self.cal_window_end),
            'cal_rows': self.cal_rows,
            'created_at': str(self.created_at),
            'model_path': self.model_path,
            'dataset': self.dataset,
            'run_name': self.run_name,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> BinMetadata:
        """Load from dictionary."""
        return cls(**d)


def get_bin_path(
    bins_root: str | Path,
    dataset: str,
    run_name: str,
    bin_month: str,
) -> Path:
    """Get the file path for a bin metadata file.

    Args:
        bins_root: Root directory for bin files
        dataset: Dataset identifier
        run_name: Run name
        bin_month: Month string (YYYY-MM)

    Returns:
        Path to .pkl file
    """
    bins_root = Path(bins_root)
    return bins_root / dataset / run_name / f"{bin_month}.pkl"


def save_bin_metadata(
    metadata: BinMetadata,
    bins_root: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Save bin metadata to filesystem.

    Args:
        metadata: BinMetadata object to save
        bins_root: Root directory for bin files
        overwrite: If True, overwrite existing file

    Returns:
        Path to saved file

    Raises:
        FileExistsError: If file exists and overwrite=False
    """
    path = get_bin_path(bins_root, metadata.dataset, metadata.run_name, metadata.bin_month)

    if path.exists() and not overwrite:
        raise FileExistsError(f"Bin metadata already exists: {path}. Use overwrite=True to replace.")

    # Create parent directories
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save as pickle
    with open(path, 'wb') as f:
        pickle.dump(metadata.to_dict(), f, protocol=pickle.HIGHEST_PROTOCOL)

    return path


def load_bin_metadata(
    bins_root: str | Path,
    dataset: str,
    run_name: str,
    bin_month: str,
) -> BinMetadata:
    """Load bin metadata from filesystem.

    Args:
        bins_root: Root directory for bin files
        dataset: Dataset identifier
        run_name: Run name
        bin_month: Month string (YYYY-MM)

    Returns:
        BinMetadata object

    Raises:
        FileNotFoundError: If bin file doesn't exist
    """
    path = get_bin_path(bins_root, dataset, run_name, bin_month)

    if not path.exists():
        raise FileNotFoundError(f"Bin metadata not found: {path}")

    with open(path, 'rb') as f:
        data = pickle.load(f)

    return BinMetadata.from_dict(data)


def list_available_bins(
    bins_root: str | Path,
    dataset: Optional[str] = None,
    run_name: Optional[str] = None,
) -> List[Dict[str, str]]:
    """List all available bin metadata files.

    Args:
        bins_root: Root directory for bin files
        dataset: Optional filter by dataset
        run_name: Optional filter by run_name

    Returns:
        List of dicts with keys: dataset, run_name, bin_month, path
    """
    bins_root = Path(bins_root)

    if not bins_root.exists():
        return []

    results = []

    # Pattern: bins/{dataset}/{run_name}/{YYYY-MM}.pkl
    pattern = "**/*.pkl"

    for pkl_file in bins_root.glob(pattern):
        # Extract dataset and run_name from path
        parts = pkl_file.relative_to(bins_root).parts
        if len(parts) != 3:
            continue

        ds, rn, month_file = parts

        # Apply filters
        if dataset and ds != dataset:
            continue
        if run_name and rn != run_name:
            continue

        bin_month = month_file.replace('.pkl', '')

        results.append({
            'dataset': ds,
            'run_name': rn,
            'bin_month': bin_month,
            'path': str(pkl_file),
        })

    return sorted(results, key=lambda x: (x['dataset'], x['run_name'], x['bin_month']))


__all__ = [
    'BinMetadata',
    'get_bin_path',
    'save_bin_metadata',
    'load_bin_metadata',
    'list_available_bins',
]
