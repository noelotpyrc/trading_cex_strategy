"""Streamlit dashboard for visualising BTCUSDT perp predictions and features."""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_loader import (
    DataLoadError,
    PREDICTION_HORIZON_HOURS,
    compute_signal_metrics,
    load_feature_frame,
    load_feature_keys,
    load_ohlcv,
    load_predictions,
)


FALLBACK_OHLCV_DB = Path("/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_ohlcv.duckdb")
FALLBACK_PREDICTIONS_DB = Path("/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_prediction.duckdb")
FALLBACK_FEATURES_DB = Path("/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_feature.duckdb")


def _resolve_default_path(env_var: str, fallback: Path) -> str:
    env_val = os.environ.get(env_var)
    return env_val if env_val else str(fallback)


DEFAULT_OHLCV_DB = _resolve_default_path("DASHBOARD_OHLCV_DUCKDB", FALLBACK_OHLCV_DB)
DEFAULT_PREDICTIONS_DB = _resolve_default_path("DASHBOARD_PREDICTIONS_DUCKDB", FALLBACK_PREDICTIONS_DB)
DEFAULT_FEATURES_DB = _resolve_default_path("DASHBOARD_FEATURES_DUCKDB", FALLBACK_FEATURES_DB)


@st.cache_data(ttl=None, show_spinner=False)
def cached_load_ohlcv(path_str: str) -> pd.DataFrame:
    if not path_str:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    return load_ohlcv(Path(path_str))


@st.cache_data(ttl=None, show_spinner=False)
def cached_load_predictions(path_str: str) -> pd.DataFrame:
    if not path_str:
        return pd.DataFrame(columns=["timestamp", "y_pred", "model_path", "feature_key", "created_at"])
    return load_predictions(Path(path_str))


@st.cache_data(ttl=None, show_spinner=False)
def cached_load_feature_keys(path_str: str) -> Iterable[str]:
    if not path_str:
        return []
    return load_feature_keys(Path(path_str))


@st.cache_data(ttl=None, show_spinner=False)
def cached_load_feature_frame(path_str: str, feature_key: str) -> pd.DataFrame:
    if not path_str or not feature_key:
        return pd.DataFrame(columns=["timestamp"])
    return load_feature_frame(Path(path_str), feature_key=feature_key)


@st.cache_data(ttl=None, show_spinner=False)
def cached_load_feature_importance(model_path: str) -> pd.DataFrame:
    if not model_path:
        return pd.DataFrame(columns=["feature", "importance", "__source_path__"])

    model_file = Path(model_path)
    candidate_paths: List[Path] = []
    if model_file.is_file():
        candidate_paths.append(model_file.parent / "feature_importance.csv")
    if model_file.is_dir():
        candidate_paths.append(model_file / "feature_importance.csv")
    if model_file.parent != model_file:
        candidate_paths.append(model_file.parent / "feature_importance.csv")
    if model_file.parent.parent != model_file.parent:
        candidate_paths.append(model_file.parent.parent / "feature_importance.csv")

    seen: set[Path] = set()
    for cand in candidate_paths:
        if cand in seen or not cand.exists():
            continue
        seen.add(cand)
        try:
            df = pd.read_csv(cand)
        except Exception:
            continue
        if df.empty:
            continue
        df = df.copy()
        df["__source_path__"] = str(cand)
        return df

    return pd.DataFrame(columns=["feature", "importance", "__source_path__"])


def _date_range_input(default_days: int = 90) -> tuple[date, date]:
    today = datetime.utcnow().date()
    start_default = today - timedelta(days=default_days)
    date_range = st.sidebar.date_input(
        "Date range",
        value=(start_default, today),
        max_value=today,
        help="Filter charts to this inclusive date window (UTC).",
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = sorted(date_range)
    else:
        start_date = date_range if isinstance(date_range, date) else start_default
        end_date = today
    return start_date, end_date


def _filter_by_range(df: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    if df.empty or "timestamp" not in df.columns:
        return df
    mask = (df["timestamp"] >= start_ts) & (df["timestamp"] < end_ts)
    return df.loc[mask].copy()


def _build_market_figure(
    ohlcv_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    *,
    show_volume: bool = True,
) -> go.Figure:
    fig = go.Figure()

    if not ohlcv_df.empty:
        fig.add_trace(
            go.Candlestick(
                x=ohlcv_df["timestamp"],
                open=ohlcv_df["open"],
                high=ohlcv_df["high"],
                low=ohlcv_df["low"],
                close=ohlcv_df["close"],
                name="OHLCV (1h)",
                increasing_line_color="#2ca02c",
                decreasing_line_color="#d62728",
                showlegend=False,
            )
        )

        if show_volume and "volume" in ohlcv_df.columns:
            fig.add_trace(
                go.Bar(
                    x=ohlcv_df["timestamp"],
                    y=ohlcv_df["volume"],
                    name="Volume",
                    marker_color="rgba(158,202,225,0.4)",
                    yaxis="y2",
                    hovertemplate="%{x}<br>Volume=%{y:,.0f}<extra></extra>",
                )
            )

    if not signals_df.empty:
        close_series = signals_df["close"] if "close" in signals_df.columns else pd.Series(np.nan, index=signals_df.index)
        close_values = pd.to_numeric(close_series, errors="coerce")
        drawdown_series = signals_df["max_drawdown"] if "max_drawdown" in signals_df.columns else pd.Series(np.nan, index=signals_df.index)
        drawdown_raw = pd.to_numeric(drawdown_series, errors="coerce")
        color_map = {True: "#2ca02c", False: "#d62728", None: "#7f7f7f"}
        is_correct = signals_df.get("is_correct", pd.Series(dtype=object)).reindex(signals_df.index)
        colors = [color_map.get(val, "#7f7f7f") for val in is_correct]

        direction_series = signals_df.get("direction", pd.Series(dtype=object)).reindex(signals_df.index).fillna("unknown")
        symbols = direction_series.apply(
            lambda dir_flag: "triangle-up" if dir_flag == "long" else ("triangle-down" if dir_flag == "short" else "circle")
        ).tolist()

        drawdown = drawdown_raw.abs().fillna(0.0)
        opacity = 1.0 - np.clip(drawdown / 0.10, 0.0, 0.7)  # >=10% drawdown -> opacity 0.3
        opacity = opacity.clip(lower=0.3, upper=1.0)

        hover_text = []
        for _, row in signals_df.iterrows():
            direction_val = row.get("direction")
            direction = direction_val if isinstance(direction_val, str) and direction_val else "n/a"
            fwd_val = row.get("forward_return")
            dd_val = row.get("max_drawdown")
            status = "Correct" if row.get("is_correct") is True else (
                "Incorrect" if row.get("is_correct") is False else "Pending"
            )
            pred_val = row.get("y_pred")
            pred_str = "n/a" if pd.isna(pred_val) else f"{float(pred_val):.4f}"
            if pd.isna(fwd_val):
                fwd_str = "168h return: n/a"
            else:
                fwd_str = f"168h return: {float(fwd_val)*100:.2f}%"
            if pd.isna(dd_val):
                dd_str = "Max drawdown: n/a"
            else:
                dd_str = f"Max drawdown: {float(dd_val)*100:.2f}%"
            meta_lines = [
                f"Direction: {direction}",
                f"Prediction: {pred_str}",
            ]
            meta_lines.append(fwd_str)
            meta_lines.append(dd_str)
            meta_lines.append(f"Status: {status}")
            model_path = row.get("model_path")
            if isinstance(model_path, str) and model_path:
                meta_lines.append(model_path)
            hover_text.append("<br>".join(meta_lines))

        fig.add_trace(
            go.Scatter(
                x=signals_df["timestamp"],
                y=close_values,
                mode="markers",
                marker=dict(
                    size=10,
                    color=colors,
                    opacity=opacity,
                    line=dict(color="#1f2937", width=1),
                    symbol=symbols,
                ),
                name="Signals",
                hovertext=hover_text,
                hovertemplate="%{x}<br>%{hovertext}<extra></extra>",
            )
        )

    layout_kwargs = dict(
        height=600,
        margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(title="Timestamp", rangeslider=dict(visible=False)),
        yaxis=dict(title="Price"),
        hovermode="x unified",
        template="plotly_white",
    )

    if show_volume and "volume" in ohlcv_df.columns:
        layout_kwargs["yaxis2"] = dict(
            title="Volume",
            overlaying="y",
            side="right",
            showgrid=False,
            rangemode="tozero",
        )

    fig.update_layout(**layout_kwargs)
    return fig


def _build_feature_figure(feature_df: pd.DataFrame, feature_name: str) -> go.Figure:
    fig = go.Figure()

    if feature_name and feature_name in feature_df.columns:
        fig.add_trace(
            go.Scatter(
                x=feature_df["timestamp"],
                y=feature_df[feature_name],
                mode="lines",
                name=feature_name,
                line=dict(color="#1f77b4"),
                hovertemplate="%{x}<br>%{y:.6f}<extra></extra>",
            )
        )

    fig.update_layout(
        height=320,
        margin=dict(l=0, r=0, t=20, b=30),
        xaxis=dict(title="Timestamp"),
        yaxis=dict(title=feature_name or "Feature value"),
        template="plotly_white",
    )
    return fig


def _build_rolling_correlation_figure(corr_df: pd.DataFrame, window_size: int) -> go.Figure:
    fig = go.Figure()

    if not corr_df.empty:
        fig.add_trace(
            go.Scatter(
                x=corr_df["timestamp"],
                y=corr_df["rolling_corr"],
                mode="lines",
                name=f"Rolling correlation ({window_size})",
                line=dict(color="#ef553b"),
                hovertemplate="%{x}<br>r=%{y:.3f}<extra></extra>",
            )
        )

    fig.update_layout(
        height=320,
        margin=dict(l=0, r=0, t=20, b=30),
        xaxis=dict(title="Timestamp"),
        yaxis=dict(title="Pearson r", range=[-1.05, 1.05]),
        template="plotly_white",
        shapes=[
            dict(
                type="line",
                xref="paper",
                x0=0,
                x1=1,
                y0=0,
                y1=0,
                line=dict(color="#9ca3af", dash="dash"),
            )
        ],
    )
    return fig


def _timestamp_bounds(start_date: date, end_date: date) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_ts = pd.Timestamp(start_date).floor("D")
    end_ts = pd.Timestamp(end_date + timedelta(days=1)).floor("D")
    return start_ts, end_ts


def _show_data_load_error(label: str, error: Exception) -> None:
    st.error(f"Failed to load {label}: {error}")


def _resolve_model_path(predictions_df: pd.DataFrame) -> str:
    if predictions_df.empty or "model_path" not in predictions_df.columns:
        return ""
    non_null = predictions_df.dropna(subset=["model_path"]).copy()
    if non_null.empty:
        return ""
    if "timestamp" in non_null.columns:
        non_null = non_null.sort_values("timestamp")
    else:
        non_null = non_null.sort_index()
    last_path = non_null["model_path"].iloc[-1]
    return str(last_path) if isinstance(last_path, str) else ""


def _normalise_importance_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    col_map = {c.lower(): c for c in df.columns}
    feature_col = col_map.get("feature") or next((c for c in df.columns if "feature" in c.lower()), None)
    importance_col = col_map.get("importance") or next((c for c in df.columns if "importance" in c.lower() or "gain" in c.lower()), None)
    if not feature_col or not importance_col:
        return pd.DataFrame(columns=["feature", "importance"])
    cleaned = df[[feature_col, importance_col]].rename(columns={feature_col: "feature", importance_col: "importance"})
    cleaned = cleaned.dropna(subset=["feature"]).copy()
    cleaned["importance"] = pd.to_numeric(cleaned["importance"], errors="coerce")
    cleaned = cleaned.dropna(subset=["importance"])
    cleaned = cleaned.sort_values("importance", ascending=False)
    return cleaned


def _select_feature_options(
    available_features: List[str],
    importance_df: pd.DataFrame,
    *,
    top_n: int = 15,
) -> List[str]:
    if not available_features:
        return []
    if importance_df.empty:
        return available_features

    feature_order = [feat for feat in importance_df["feature"].tolist() if feat in available_features]
    if not feature_order:
        return available_features

    limited = feature_order[:top_n]
    remainder = [feat for feat in feature_order[top_n:] if feat not in limited]
    others = [feat for feat in available_features if feat not in limited and feat not in remainder]

    ordered = limited + remainder + others
    seen: set[str] = set()
    deduped = []
    for feat in ordered:
        if feat in seen:
            continue
        seen.add(feat)
        deduped.append(feat)
    return deduped


def main() -> None:
    st.set_page_config(page_title="BTCUSDT Signals Dashboard", layout="wide")
    st.title("BTCUSDT 1h Prediction Dashboard")

    with st.sidebar:
        st.header("Controls")
        ohlcv_path = st.text_input(
            "OHLCV DuckDB path",
            value=DEFAULT_OHLCV_DB,
            placeholder="/path/to/ohlcv.duckdb",
        )
        predictions_path = st.text_input(
            "Predictions DuckDB path",
            value=DEFAULT_PREDICTIONS_DB,
            placeholder="/path/to/binance_btcusdt_perp_prediction.duckdb",
        )
        features_path = st.text_input(
            "Features DuckDB path",
            value=DEFAULT_FEATURES_DB,
            placeholder="/path/to/binance_btcusdt_perp_feature.duckdb",
        )

        refresh_clicked = st.button("Refresh data", type="primary", help="Clear cached queries and reload from DuckDB")
        if refresh_clicked:
            st.cache_data.clear()
            rerun_fn = getattr(st, "rerun", None)
            if callable(rerun_fn):
                rerun_fn()
            else:
                legacy_rerun = getattr(st, "experimental_rerun", None)
                if callable(legacy_rerun):
                    legacy_rerun()
                else:
                    st.warning("Please refresh the page to reload updated data.")

        start_date, end_date = _date_range_input()
        direction_threshold = st.text_input(
            "Prediction threshold",
            value="0.00",
            help="Signals > threshold mark long entries; signals < threshold mark short entries.",
        )
        try:
            threshold_value = float(direction_threshold)
        except (TypeError, ValueError):
            st.warning("Invalid threshold input. Using 0.0")
            threshold_value = 0.0
        show_long = st.checkbox(
            "Show long signals",
            value=True,
            help="Display predictions where y_pred exceeds the threshold.",
        )
        show_short = st.checkbox(
            "Show short signals",
            value=False,
            help="Display predictions where y_pred falls below the threshold.",
        )
        show_volume = st.checkbox(
            "Show volume",
            value=True,
            help="Toggle the 1h volume overlay on the candlestick chart.",
        )
        rolling_correlation_window = st.number_input(
            "Rolling correlation window",
            min_value=20,
            max_value=2000,
            value=50,
            step=10,
            help="Number of samples used for the rolling correlation time series.",
        )
        correlation_window = st.number_input(
            "Correlation window (rows)",
            min_value=50,
            max_value=10000,
            value=500,
            step=50,
            help="Number of most recent signals used for the y_pred vs 168h log return correlation.",
        )

    start_ts, end_ts = _timestamp_bounds(start_date, end_date)
    rolling_correlation_window = int(rolling_correlation_window)
    correlation_window = int(correlation_window)

    try:
        ohlcv_df = cached_load_ohlcv(ohlcv_path)
    except DataLoadError as exc:
        _show_data_load_error("OHLCV", exc)
        ohlcv_df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    try:
        predictions_df = cached_load_predictions(predictions_path)
    except DataLoadError as exc:
        _show_data_load_error("predictions", exc)
        predictions_df = pd.DataFrame(columns=["timestamp", "y_pred", "model_path", "feature_key", "created_at"])

    feature_keys: Iterable[str] = []
    try:
        feature_keys = cached_load_feature_keys(features_path)
    except DataLoadError as exc:
        _show_data_load_error("feature keys", exc)

    feature_key_default = next(iter(feature_keys), "")
    with st.sidebar:
        st.subheader("Feature panel")
        selected_feature_key = st.selectbox(
            "Feature key",
            options=list(feature_keys) if feature_keys else [feature_key_default] if feature_key_default else [],
            index=0,
            disabled=not feature_keys,
        )

    feature_df = pd.DataFrame(columns=["timestamp"])
    if selected_feature_key:
        try:
            feature_df = cached_load_feature_frame(features_path, selected_feature_key)
        except DataLoadError as exc:
            _show_data_load_error("features", exc)

    model_path_for_importance = _resolve_model_path(predictions_df)
    importance_raw = cached_load_feature_importance(model_path_for_importance) if model_path_for_importance else pd.DataFrame()
    importance_df = _normalise_importance_columns(importance_raw)

    feature_columns = [col for col in feature_df.columns if col != "timestamp"]
    feature_options = _select_feature_options(feature_columns, importance_df)
    feature_name_default = feature_options[0] if feature_options else ""
    with st.sidebar:
        if feature_options:
            selected_feature = st.selectbox(
                "Feature",
                options=feature_options,
                index=0,
            )
        else:
            selected_feature = ""
            st.selectbox(
                "Feature",
                options=["No features available"],
                index=0,
                disabled=True,
            )
        if not importance_df.empty and model_path_for_importance:
            importance_source = None
            if "__source_path__" in importance_raw.columns:
                source_col = importance_raw["__source_path__"].dropna()
                if not source_col.empty:
                    importance_source = source_col.iloc[0]
            if importance_source:
                st.caption(f"Feature importance source: {importance_source}")

    signals_with_metrics = compute_signal_metrics(
        predictions_df,
        ohlcv_df,
        horizon_hours=PREDICTION_HORIZON_HOURS,
        direction_threshold=threshold_value,
        include_long=show_long,
        include_short=show_short,
    )

    correlation_source = compute_signal_metrics(
        predictions_df,
        ohlcv_df,
        horizon_hours=PREDICTION_HORIZON_HOURS,
        direction_threshold=float("-inf"),
        include_long=True,
        include_short=False,
    )

    correlation_value = np.nan
    correlation_sample_size = 0
    correlation_window_start: pd.Timestamp | None = None
    correlation_window_end: pd.Timestamp | None = None
    rolling_correlation_df = pd.DataFrame(columns=["timestamp", "rolling_corr"])
    if not correlation_source.empty:
        correlation_frame = (
            correlation_source[["timestamp", "y_pred", "forward_log_return"]]
            .dropna(subset=["y_pred", "forward_log_return"])
            .sort_values("timestamp")
        )
        if not correlation_frame.empty:
            recent_signals = correlation_frame.tail(correlation_window)
            correlation_sample_size = len(recent_signals)
            if correlation_sample_size >= 1:
                correlation_window_start = recent_signals["timestamp"].iloc[0]
                correlation_window_end = recent_signals["timestamp"].iloc[-1]
            if correlation_sample_size >= 2:
                corr = recent_signals["y_pred"].corr(recent_signals["forward_log_return"])
                if pd.notna(corr):
                    correlation_value = float(corr)

            min_periods = max(10, rolling_correlation_window // 2)
            rolling_series = (
                correlation_frame["y_pred"]
                .rolling(window=rolling_correlation_window, min_periods=min_periods)
                .corr(correlation_frame["forward_log_return"])
            )
            rolling_correlation_df = correlation_frame[["timestamp"]].copy()
            rolling_correlation_df["rolling_corr"] = rolling_series
            rolling_correlation_df = rolling_correlation_df.dropna(subset=["rolling_corr"])

    with st.sidebar:
        st.subheader("Signal correlation")
        if correlation_sample_size >= 2 and np.isfinite(correlation_value):
            st.metric("Pearson r (y_pred vs 168h log)", f"{correlation_value:.3f}")
            caption_lines = [f"Rows used: {correlation_sample_size} (window: {correlation_window})"]
            if correlation_window_start is not None and correlation_window_end is not None:
                start_str = correlation_window_start.strftime("%Y-%m-%d %H:%M")
                end_str = correlation_window_end.strftime("%Y-%m-%d %H:%M")
                caption_lines.append(f"Range: {start_str} → {end_str} UTC")
            st.caption(" | ".join(caption_lines))
        else:
            st.metric("Pearson r (y_pred vs 168h log)", "n/a")
            st.caption("Not enough realized signals to compute correlation.")

    rolling_correlation_display = _filter_by_range(rolling_correlation_df, start_ts, end_ts)
    if not rolling_correlation_display.empty:
        rolling_corr_fig = _build_rolling_correlation_figure(rolling_correlation_display, rolling_correlation_window)
        st.plotly_chart(rolling_corr_fig, use_container_width=True)
    elif not rolling_correlation_df.empty:
        st.caption("No rolling correlation points inside the selected date range.")
    else:
        st.caption(
            "Rolling correlation requires sufficient realized signals; adjust the window or ensure data is available."
        )

    ohlcv_display = _filter_by_range(ohlcv_df, start_ts, end_ts)
    signals_display = _filter_by_range(signals_with_metrics, start_ts, end_ts)
    feature_display = _filter_by_range(feature_df, start_ts, end_ts)

    if ohlcv_display.empty:
        st.warning("No OHLCV rows available for the selected range. Adjust the date filter or verify the DuckDB path.")
    else:
        market_fig = _build_market_figure(ohlcv_display, signals_display, show_volume=show_volume)
        st.plotly_chart(market_fig, use_container_width=True)

    st.divider()

    if feature_display.empty or not selected_feature:
        st.info("Select a feature key and column to view the feature time series.")
    else:
        feature_fig = _build_feature_figure(feature_display, selected_feature)
        st.plotly_chart(feature_fig, use_container_width=True)

    st.caption(
        "Signals assessed using a 168h forward return window and max drawdown based on OHLC lows/highs. "
        "Use the refresh button after running new backfills."
    )


if __name__ == "__main__":
    main()
