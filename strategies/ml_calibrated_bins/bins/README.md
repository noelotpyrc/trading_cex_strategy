# Bin Metadata Storage

This directory stores monthly bin metadata files containing calibrators and bin edges.

## Structure

```
bins/
└── {dataset}/                    # e.g., binance_btcusdt_perp_1h
    └── {run_name}/               # e.g., run_20251102_231428_lgbm_...
        ├── 2025-05.pkl          # May 2025 bins
        ├── 2025-06.pkl          # June 2025 bins
        └── 2025-07.pkl          # July 2025 bins
```

## File Contents

Each `.pkl` file contains a dictionary with:
- `bin_month`: Month identifier (YYYY-MM)
- `n_bins`: Number of bins (e.g., 20)
- `calibration_method`: 'platt' or 'isotonic'
- `bin_edges`: Numpy array of quantile edges
- `calibrator`: Fitted calibrator object
- `cal_window_start`: Start of calibration window
- `cal_window_end`: End of calibration window
- `cal_rows`: Number of samples in calibration window
- `created_at`: Creation timestamp
- `model_path`: Full path to model file
- `dataset`: Dataset identifier
- `run_name`: Run name

## Size

Each file is typically 50-200 KB (very lightweight).

## Version Control

You can track these files in git for reproducibility. If you have many models/months and want to save space, you can `.gitignore` this directory and store bins separately.
