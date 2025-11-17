#!/usr/bin/env python3
"""
Quick script to inspect bin metadata files.

Usage:
    python strategies/ml_calibrated_bins/inspect_bins.py 2025-05
    python strategies/ml_calibrated_bins/inspect_bins.py 2025-05 --test
"""
import argparse
import sys
from pathlib import Path

import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from strategies.ml_calibrated_bins.calibration.bin_manager import load_bin_metadata


def inspect_bin_file(
    bin_month: str,
    dataset: str = 'binance_btcusdt_perp_1h',
    run_name: str = 'run_20251102_231428_lgbm_y_tp_before_sl_u0.04_d0.02_24h_binary',
    bins_root: str = './strategies/ml_calibrated_bins/bins',
    test_calibrator: bool = False,
):
    """Inspect a bin metadata file."""

    # Load bin metadata
    bin_meta = load_bin_metadata(
        bins_root=bins_root,
        dataset=dataset,
        run_name=run_name,
        bin_month=bin_month
    )

    print(f'\n{"="*70}')
    print(f'  BIN METADATA: {bin_month}')
    print(f'{"="*70}\n')

    # Basic info
    print(f'📅 Bin Month:          {bin_meta.bin_month}')
    print(f'📊 Dataset:            {bin_meta.dataset}')
    print(f'🏃 Run Name:           {bin_meta.run_name}')
    print(f'🔧 Calibration Method: {bin_meta.calibration_method}')
    print(f'🗓️  Created At:         {bin_meta.created_at}')
    print()

    # Calibration window
    print(f'📈 Calibration Window:')
    print(f'   Start:  {bin_meta.cal_window_start}')
    print(f'   End:    {bin_meta.cal_window_end}')
    print(f'   Rows:   {bin_meta.cal_rows:,}')
    print()

    # Bin edges
    print(f'🎯 Bin Configuration ({bin_meta.n_bins} bins, {len(bin_meta.bin_edges)} edges):')
    print()
    print(f'  {"Bin":>4} {"Probability Range":>30} {"Percentile":>12}')
    print(f'  {"-"*4} {"-"*30} {"-"*12}')

    # First bin
    edge = bin_meta.bin_edges[0]
    print(f'  {1:>4} {"0.000000":>13} - {edge:>13.6f}  {"  0 -   5%":>12}')

    # Middle bins
    for i in range(len(bin_meta.bin_edges) - 1):
        bin_num = i + 2
        edge_low = bin_meta.bin_edges[i]
        edge_high = bin_meta.bin_edges[i + 1]
        pct_low = (bin_num - 1) * 5
        pct_high = bin_num * 5
        print(f'  {bin_num:>4} {edge_low:>13.6f} - {edge_high:>13.6f}  {f"{pct_low:3d} - {pct_high:3d}%":>12}')

    # Last bin
    edge = bin_meta.bin_edges[-1]
    print(f'  {bin_meta.n_bins:>4} {edge:>13.6f} - {"1.000000":>13}  {" 95 - 100%":>12}')
    print()

    # Model info
    print(f'📁 Model Path:')
    print(f'   {bin_meta.model_path}')
    print()

    # Test calibrator if requested
    if test_calibrator:
        print(f'{"="*70}')
        print(f'  CALIBRATOR TEST')
        print(f'{"="*70}\n')

        test_preds = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
        calibrated_probs = bin_meta.calibrator.predict_proba(test_preds)
        bins = np.digitize(calibrated_probs, bin_meta.bin_edges) + 1

        print(f'  {"Raw Pred":>10}  {"→":^3}  {"Calibrated Prob":>16}  {"→":^3}  {"Bin":>4}')
        print(f'  {"-"*10}  {"-"*3}  {"-"*16}  {"-"*3}  {"-"*4}')

        for raw, cal, b in zip(test_preds, calibrated_probs, bins):
            print(f'  {raw:>10.2f}  {"→":^3}  {cal:>10.6f} ({cal*100:>4.1f}%)  {"→":^3}  {b:>4}')
        print()

    print(f'{"="*70}\n')


def main():
    parser = argparse.ArgumentParser(description='Inspect bin metadata files')
    parser.add_argument('bin_month', help='Bin month (YYYY-MM)')
    parser.add_argument('--dataset', default='binance_btcusdt_perp_1h', help='Dataset identifier')
    parser.add_argument('--run-name', default='run_20251102_231428_lgbm_y_tp_before_sl_u0.04_d0.02_24h_binary',
                        help='Run name')
    parser.add_argument('--bins-root', default='./strategies/ml_calibrated_bins/bins', help='Bins root directory')
    parser.add_argument('--test', action='store_true', help='Test calibrator with sample predictions')

    args = parser.parse_args()

    inspect_bin_file(
        bin_month=args.bin_month,
        dataset=args.dataset,
        run_name=args.run_name,
        bins_root=args.bins_root,
        test_calibrator=args.test,
    )


if __name__ == '__main__':
    main()
