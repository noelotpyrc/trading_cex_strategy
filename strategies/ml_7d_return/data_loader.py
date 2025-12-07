"""Data access and preparation utilities for the Streamlit dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import duckdb  # type: ignore
import numpy as np
import pandas as pd


DEFAULT_OHLCV_TABLE = "ohlcv_btcusdt_1h"
DEFAULT_PREDICTIONS_TABLE = "predictions"
DEFAULT_FEATURES_TABLE = "features"
PREDICTION_HORIZON_HOURS = 168


class DataLoadError(RuntimeError):
    """Raised when a required data artifact cannot be loaded."""

def _check_db_exists(path: Path) -> None:
    if not path.exists():
        raise DataLoadError(f"DuckDB file not found: {path}")


def _connect(path: Path) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(path))
    con.execute("SET TimeZone='UTC';")
    return con


def load_ohlcv(
    db_path: Path,
    *,
    table: str = DEFAULT_OHLCV_TABLE,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Load OHLCV candles from DuckDB, normalizing timestamps and column names."""

    _check_db_exists(db_path)

    clauses: List[str] = []
    params: List[object] = []
    if start is not None:
        clauses.append("timestamp >= ?")
        params.append(pd.Timestamp(start).to_pydatetime())
    if end is not None:
        clauses.append("timestamp <= ?")
        params.append(pd.Timestamp(end).to_pydatetime())

    where_clause = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    query = (
        f"SELECT timestamp, open, high, low, close, volume "
        f"FROM {table}{where_clause} ORDER BY timestamp ASC"
    )

    with _connect(db_path) as con:
        df = con.execute(query, params).fetch_df()

    if df.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df.columns = [str(c).strip().lower() for c in df.columns]
    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["timestamp"] = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def load_predictions(
    db_path: Path,
    *,
    table: str = DEFAULT_PREDICTIONS_TABLE,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Load model predictions stored in DuckDB."""

    _check_db_exists(db_path)

    clauses: List[str] = []
    params: List[object] = []
    if start is not None:
        clauses.append("ts >= ?")
        params.append(pd.Timestamp(start).to_pydatetime())
    if end is not None:
        clauses.append("ts <= ?")
        params.append(pd.Timestamp(end).to_pydatetime())

    where_clause = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    query = (
        f"SELECT ts AS timestamp, y_pred, model_path, feature_key, created_at "
        f"FROM {table}{where_clause} ORDER BY timestamp ASC"
    )

    with _connect(db_path) as con:
        df = con.execute(query, params).fetch_df()

    if df.empty:
        return pd.DataFrame(columns=["timestamp", "y_pred", "model_path", "feature_key", "created_at"])

    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["timestamp"] = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True).dt.tz_localize(None)
    df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def load_feature_keys(
    db_path: Path,
    *,
    table: str = DEFAULT_FEATURES_TABLE,
) -> List[str]:
    """Return available feature_key values from the features table."""

    _check_db_exists(db_path)
    query = f"SELECT DISTINCT feature_key FROM {table} ORDER BY feature_key"
    with _connect(db_path) as con:
        rows = con.execute(query).fetchall()
    return [row[0] for row in rows]


def load_feature_frame(
    db_path: Path,
    *,
    feature_key: str,
    table: str = DEFAULT_FEATURES_TABLE,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Load feature snapshots for a given key, expanding the JSON column."""

    _check_db_exists(db_path)

    clauses = ["feature_key = ?"]
    params: List[object] = [feature_key]
    if start is not None:
        clauses.append("ts >= ?")
        params.append(pd.Timestamp(start).to_pydatetime())
    if end is not None:
        clauses.append("ts <= ?")
        params.append(pd.Timestamp(end).to_pydatetime())

    where_clause = f" WHERE {' AND '.join(clauses)}"
    query = f"SELECT ts AS timestamp, features FROM {table}{where_clause} ORDER BY timestamp ASC"

    with _connect(db_path) as con:
        df = con.execute(query, params).fetch_df()

    if df.empty:
        return pd.DataFrame(columns=["timestamp"])

    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["timestamp"] = ts.dt.tz_convert("UTC").dt.tz_localize(None)

    feature_records = []
    for raw in df["features"].tolist():
        try:
            feature_records.append(json.loads(raw))
        except (TypeError, json.JSONDecodeError):
            feature_records.append({})

    feature_matrix = pd.json_normalize(feature_records)
    feature_matrix = feature_matrix.apply(pd.to_numeric, errors="coerce")
    feature_matrix.insert(0, "timestamp", df["timestamp"].to_numpy())
    feature_matrix = feature_matrix.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return feature_matrix


def compute_signal_metrics(
    predictions: pd.DataFrame,
    ohlcv: pd.DataFrame,
    *,
    horizon_hours: int = PREDICTION_HORIZON_HOURS,
    direction_threshold: float = 0.0,
    include_long: bool = True,
    include_short: bool = False,
) -> pd.DataFrame:
    """Augment prediction rows with market-based evaluation metrics."""

    if predictions.empty or "y_pred" not in predictions.columns:
        return predictions.assign(
            direction=pd.Series(dtype="object"),
            forward_return=pd.Series(dtype="float64"),
            forward_log_return=pd.Series(dtype="float64"),
            max_drawdown=pd.Series(dtype="float64"),
            is_correct=pd.Series(dtype="object"),
            has_full_horizon=pd.Series(dtype="bool"),
        )

    numeric_preds = pd.to_numeric(predictions["y_pred"], errors="coerce")
    if not include_long and not include_short:
        return predictions.iloc[0:0].assign(
            direction=pd.Series(dtype="object"),
            forward_return=pd.Series(dtype="float64"),
            forward_log_return=pd.Series(dtype="float64"),
            max_drawdown=pd.Series(dtype="float64"),
            is_correct=pd.Series(dtype="object"),
            has_full_horizon=pd.Series(dtype="bool"),
        )

    long_mask = (numeric_preds > direction_threshold) if include_long else pd.Series(False, index=predictions.index)
    short_mask = (numeric_preds < direction_threshold) if include_short else pd.Series(False, index=predictions.index)
    long_mask = long_mask.fillna(False)
    short_mask = short_mask.fillna(False)
    combined_mask = long_mask | short_mask

    filtered = predictions.loc[combined_mask].copy()

    if filtered.empty:
        return filtered.assign(
            direction=pd.Series(dtype="object"),
            forward_return=pd.Series(dtype="float64"),
            forward_log_return=pd.Series(dtype="float64"),
            max_drawdown=pd.Series(dtype="float64"),
            is_correct=pd.Series(dtype="object"),
            has_full_horizon=pd.Series(dtype="bool"),
        )

    filtered["y_pred"] = numeric_preds.loc[filtered.index]
    filtered["direction"] = ["long" if long_mask.loc[idx] else "short" for idx in filtered.index]

    if ohlcv.empty:
        preds = filtered.copy()
        preds["forward_return"] = pd.NA
        preds["forward_log_return"] = pd.NA
        preds["max_drawdown"] = pd.NA
        preds["is_correct"] = pd.NA
        preds["has_full_horizon"] = False
        return preds

    ohlcv_sorted = ohlcv.sort_values("timestamp").reset_index(drop=True)

    close = ohlcv_sorted["close"].astype(float)
    low = ohlcv_sorted["low"].astype(float)
    high = ohlcv_sorted["high"].astype(float)

    future_close = close.shift(-horizon_hours)
    min_future_low = low.iloc[::-1].rolling(window=horizon_hours, min_periods=1).min().iloc[::-1].shift(-1)
    max_future_high = high.iloc[::-1].rolling(window=horizon_hours, min_periods=1).max().iloc[::-1].shift(-1)

    metrics_frame = ohlcv_sorted[["timestamp", "close"]].copy()
    metrics_frame["future_close"] = future_close
    metrics_frame["min_future_low"] = min_future_low
    metrics_frame["max_future_high"] = max_future_high
    with np.errstate(divide="ignore", invalid="ignore"):
        close_positive = (metrics_frame["close"] > 0) & (metrics_frame["future_close"] > 0)
        log_ret = np.where(
            close_positive & metrics_frame["future_close"].notna(),
            np.log(metrics_frame["future_close"]) - np.log(metrics_frame["close"]),
            np.nan,
        )
    metrics_frame["forward_log_return"] = log_ret

    merged = filtered.merge(metrics_frame, on="timestamp", how="left", copy=False, suffixes=("", "_mkt"))

    merged["has_full_horizon"] = merged["future_close"].notna()

    merged["forward_return"] = (
        (merged["future_close"] / merged["close"]) - 1.0
    ).where(merged["has_full_horizon"])

    merged["forward_log_return"] = merged["forward_log_return"].where(merged["has_full_horizon"])

    def _drawdown(row: pd.Series) -> float:
        entry = row.get("close")
        if pd.isna(entry):
            return pd.NA
        direction = row.get("direction")
        if direction == "long":
            low_val = row.get("min_future_low")
            if pd.isna(low_val):
                return pd.NA
            return (low_val / entry) - 1.0
        # short direction
        high_val = row.get("max_future_high")
        if pd.isna(high_val):
            return pd.NA
        return (entry / high_val) - 1.0

    merged["max_drawdown"] = merged.apply(_drawdown, axis=1)

    def _correct(row: pd.Series) -> Optional[bool]:
        fwd = row.get("forward_return")
        if pd.isna(fwd):
            return None
        if row["direction"] == "long":
            return bool(fwd > 0)
        return bool(fwd < 0)

    merged["is_correct"] = merged.apply(_correct, axis=1)
    return merged


def clamp_series(series: pd.Series, *, min_value: float, max_value: float) -> pd.Series:
    """Utility to clamp numeric series within bounds (used for marker opacity)."""

    return series.clip(lower=min_value, upper=max_value)
