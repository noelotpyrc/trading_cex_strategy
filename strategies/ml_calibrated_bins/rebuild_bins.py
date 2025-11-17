#!/usr/bin/env python3
"""
Monthly bin rebuilding script for ML Calibrated Bins strategy.

Fetches predictions and actuals from DuckDB, fits calibrator on rolling 12-month
window, derives quantile bin edges, and saves bin metadata to filesystem.

Usage:
    python strategies/ml_calibrated_bins/rebuild_bins.py \\
        --pred-duckdb /path/to/predictions.duckdb \\
        --ohlcv-duckdb /path/to/ohlcv.duckdb \\
        --model-path /path/to/model.txt \\
        --dataset binance_btcusdt_perp_1h \\
        --bins-root ./strategies/ml_calibrated_bins/bins \\
        --bin-month 2025-11 \\
        --lookback-months 12 \\
        --n-bins 20 \\
        --method platt

Backfill multiple months:
    python strategies/ml_calibrated_bins/rebuild_bins.py ... --bin-month 2025-05 2025-06 2025-07
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import duckdb
import numpy as np
import pandas as pd

from .calibration.bin_manager import BinMetadata, save_bin_metadata
from .calibration.calibrators import fit_calibrator, quantile_edges


def load_predictions(
    pred_duckdb: str,
    model_path: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load predictions from DuckDB for a time window, with fallback to pred_train.csv.

    First tries to load from DuckDB. If no data found, falls back to pred_train.csv
    in the model directory.

    Args:
        pred_duckdb: Path to predictions DuckDB
        model_path: Model path to filter on
        start: Start timestamp
        end: End timestamp
        verbose: Print fallback information

    Returns:
        DataFrame with columns: timestamp, y_pred
    """
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    # Try DuckDB first
    conn = duckdb.connect(pred_duckdb, read_only=True)
    query = """
        SELECT ts as timestamp, y_pred
        FROM predictions
        WHERE model_path = ?
          AND ts >= ?
          AND ts <= ?
        ORDER BY ts
    """
    df_db = conn.execute(query, [model_path, str(start), str(end)]).df()
    conn.close()

    if not df_db.empty:
        df_db['timestamp'] = pd.to_datetime(df_db['timestamp'])

    # Check if we need to fallback to CSV
    pred_train_path = Path(model_path).parent / 'pred_train.csv'

    if df_db.empty and pred_train_path.exists():
        # No DB data, use CSV only
        if verbose:
            print(f"  No predictions in DuckDB, loading from {pred_train_path.name}")
        df_csv = pd.read_csv(pred_train_path)
        df_csv['timestamp'] = pd.to_datetime(df_csv['timestamp'])
        df_csv = df_csv[(df_csv['timestamp'] >= start) & (df_csv['timestamp'] <= end)]
        df = df_csv[['timestamp', 'y_pred']].copy()
    elif not df_db.empty and pred_train_path.exists():
        # Check if we have gaps in DB data that CSV can fill
        df_csv = pd.read_csv(pred_train_path)
        df_csv['timestamp'] = pd.to_datetime(df_csv['timestamp'])
        df_csv = df_csv[(df_csv['timestamp'] >= start) & (df_csv['timestamp'] <= end)]

        # Merge, preferring DuckDB data
        df_merged = pd.concat([df_db, df_csv[['timestamp', 'y_pred']]], ignore_index=True)
        df_merged = df_merged.drop_duplicates(subset=['timestamp'], keep='first')
        df_merged = df_merged.sort_values('timestamp')

        csv_added = len(df_merged) - len(df_db)
        if csv_added > 0 and verbose:
            print(f"  Loaded {len(df_db)} predictions from DuckDB, added {csv_added} from {pred_train_path.name}")

        df = df_merged
    else:
        # Only DB data (or no CSV available)
        df = df_db

    return df


def load_actuals(
    ohlcv_duckdb: str,
    dataset: str,
    target_key: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    """Load actual targets from targets table in DuckDB.

    Note: This assumes targets have already been computed and stored in the targets table
    using the backfill_targets.py script from trading_cex_data_processing repo.

    Args:
        ohlcv_duckdb: Path to OHLCV DuckDB (contains targets table)
        dataset: Dataset identifier (e.g., 'binance_btcusdt_perp_1h')
        target_key: Target key name (e.g., 'y_tp_before_sl_u0.04_d0.02_24h')
        start: Start timestamp
        end: End timestamp

    Returns:
        DataFrame with columns: timestamp, y_true
    """
    conn = duckdb.connect(ohlcv_duckdb, read_only=True)
    query = """
        SELECT timestamp, target_value as y_true
        FROM targets
        WHERE dataset = ?
          AND target_key = ?
          AND timestamp >= ?
          AND timestamp <= ?
        ORDER BY timestamp
    """
    df = conn.execute(query, [dataset, target_key, str(start), str(end)]).df()
    conn.close()

    if df.empty:
        raise ValueError(
            f"No targets found for dataset={dataset}, target_key={target_key} "
            f"in range {start} to {end}. "
            f"Please run backfill_targets.py from trading_cex_data_processing first."
        )

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def rebuild_bins_for_month(
    pred_duckdb: str,
    ohlcv_duckdb: str,
    model_path: str,
    dataset: str,
    run_name: str,
    ohlcv_table: str,
    target_key: str,
    bin_month: str,
    lookback_months: int,
    n_bins: int,
    method: str,
    bins_root: str | Path,
    overwrite: bool = False,
    verbose: bool = True,
) -> Path:
    """Rebuild bins for a single month.

    Args:
        pred_duckdb: Path to predictions DuckDB
        ohlcv_duckdb: Path to OHLCV DuckDB
        model_path: Full path to model file
        dataset: Dataset identifier
        run_name: Run name (extracted from model_path)
        ohlcv_table: OHLCV table name
        target_key: Target column name
        bin_month: Month to rebuild (YYYY-MM)
        lookback_months: Calibration window in months
        n_bins: Number of bins
        method: Calibration method ('platt' or 'isotonic')
        bins_root: Root directory for bin files
        overwrite: If True, overwrite existing bins
        verbose: Print progress

    Returns:
        Path to saved bin file
    """
    # Parse bin_month
    month_start = pd.Timestamp(bin_month)
    month_end = (month_start + pd.offsets.MonthEnd(1)).normalize() + pd.Timedelta(hours=23)

    # Calibration window: lookback_months before month_start
    cal_start = (month_start - pd.DateOffset(months=lookback_months)).normalize()
    cal_end = month_start - pd.Timedelta(hours=1)

    if verbose:
        print(f"[{bin_month}] Rebuilding bins...")
        print(f"  Calibration window: {cal_start} to {cal_end}")
        print(f"  Validation month: {month_start} to {month_end}")

    # Load predictions for calibration window
    pred_cal = load_predictions(pred_duckdb, model_path, cal_start, cal_end, verbose=verbose)
    if verbose:
        print(f"  Loaded {len(pred_cal)} predictions for calibration")

    if pred_cal.empty:
        raise ValueError(f"No predictions found for calibration window {cal_start} to {cal_end}")

    # Load actuals for calibration window
    actuals_cal = load_actuals(ohlcv_duckdb, dataset, target_key, cal_start, cal_end)
    if verbose:
        print(f"  Loaded {len(actuals_cal)} actuals for calibration")

    # Merge predictions with actuals
    df_cal = pred_cal.merge(actuals_cal, on='timestamp', how='inner')
    df_cal = df_cal.dropna(subset=['y_pred', 'y_true'])

    if verbose:
        print(f"  Merged {len(df_cal)} rows for calibration (after dropna)")

    if df_cal.empty:
        raise ValueError("No valid data after merging predictions and actuals")

    # Fit calibrator
    if verbose:
        print(f"  Fitting {method} calibrator...")

    calibrator = fit_calibrator(df_cal['y_pred'], df_cal['y_true'], method=method)

    # Compute calibrated probabilities on calibration window
    p_cal = calibrator.predict_proba(df_cal['y_pred'])

    # Derive bin edges from quantiles
    edges = quantile_edges(p_cal, n_bins)

    if verbose:
        print(f"  Derived {len(edges)} bin edges:")
        print(f"    {np.array2string(edges, precision=4, separator=', ')}")
        print(f"  Calibrated p range: [{p_cal.min():.6f}, {p_cal.max():.6f}]")

    # Create metadata
    metadata = BinMetadata(
        bin_month=bin_month,
        n_bins=n_bins,
        calibration_method=method,
        bin_edges=edges,
        calibrator=calibrator,
        cal_window_start=str(cal_start),
        cal_window_end=str(cal_end),
        cal_rows=len(df_cal),
        created_at=str(pd.Timestamp.now()),
        model_path=model_path,
        dataset=dataset,
        run_name=run_name,
    )

    # Save to filesystem
    saved_path = save_bin_metadata(metadata, bins_root, overwrite=overwrite)

    if verbose:
        print(f"  Saved to: {saved_path}")

    return saved_path


def main():
    parser = argparse.ArgumentParser(description="Rebuild monthly bins from predictions and actuals")
    parser.add_argument('--pred-duckdb', required=True, help='Path to predictions DuckDB')
    parser.add_argument('--ohlcv-duckdb', required=True, help='Path to OHLCV DuckDB')
    parser.add_argument('--ohlcv-table', default='ohlcv_btcusdt_1h', help='OHLCV table name')
    parser.add_argument('--target-key', default='y_tp_before_sl_u0.04_d0.02_24h',
                        help='Target column name in OHLCV table')
    parser.add_argument('--model-path', required=True, help='Full path to model file')
    parser.add_argument('--dataset', required=True, help='Dataset identifier (e.g., binance_btcusdt_perp_1h)')
    parser.add_argument('--run-name', help='Run name (auto-extracted from model-path if not provided)')
    parser.add_argument('--bins-root', default='./strategies/ml_calibrated_bins/bins', help='Root directory for bin files')
    parser.add_argument('--bin-month', nargs='+', required=True,
                        help='Month(s) to rebuild (YYYY-MM), can specify multiple')
    parser.add_argument('--lookback-months', type=int, default=12,
                        help='Calibration window in months (default: 12)')
    parser.add_argument('--n-bins', type=int, default=20, help='Number of bins (default: 20)')
    parser.add_argument('--method', choices=['platt', 'isotonic'], default='platt',
                        help='Calibration method (default: platt)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing bins')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')

    args = parser.parse_args()

    # Auto-extract run_name from model_path if not provided
    run_name = args.run_name
    if not run_name:
        # Extract from path like: .../run_20251102_231428_lgbm_.../model.txt
        model_path_obj = Path(args.model_path)
        run_name = model_path_obj.parent.name
        print(f"Auto-extracted run_name: {run_name}")

    # Process each month
    for bin_month in args.bin_month:
        try:
            rebuild_bins_for_month(
                pred_duckdb=args.pred_duckdb,
                ohlcv_duckdb=args.ohlcv_duckdb,
                model_path=args.model_path,
                dataset=args.dataset,
                run_name=run_name,
                ohlcv_table=args.ohlcv_table,
                target_key=args.target_key,
                bin_month=bin_month,
                lookback_months=args.lookback_months,
                n_bins=args.n_bins,
                method=args.method,
                bins_root=args.bins_root,
                overwrite=args.overwrite,
                verbose=not args.quiet,
            )
        except Exception as e:
            print(f"ERROR: Failed to rebuild bins for {bin_month}: {e}")
            raise SystemExit(1)

    print(f"\n✓ Successfully rebuilt bins for {len(args.bin_month)} month(s)")


if __name__ == '__main__':
    main()
