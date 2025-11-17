#!/usr/bin/env python3
"""
Streamlit app for visualizing ML Calibrated Bins strategy.

Usage:
    streamlit run strategies/ml_calibrated_bins/app_bins_strategy.py
"""
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from strategies.ml_calibrated_bins.calibration import (
    load_bin_metadata,
    apply_calibration_and_bins,
)


st.set_page_config(
    page_title="ML Calibrated Bins Strategy",
    page_icon="📊",
    layout="wide",
)


@st.cache_data
def get_prediction_time_range(pred_duckdb: str, model_path: str):
    """Get the min and max timestamp from predictions table."""
    try:
        conn = duckdb.connect(pred_duckdb, read_only=True)
        query = """
            SELECT MIN(ts) as min_ts, MAX(ts) as max_ts
            FROM predictions
            WHERE model_path = ?
        """
        result = conn.execute(query, [model_path]).fetchone()
        conn.close()

        if result and result[0] and result[1]:
            return pd.to_datetime(result[0]), pd.to_datetime(result[1])
        return None, None
    except Exception:
        return None, None


@st.cache_data
def load_predictions(pred_duckdb: str, model_path: str, start: str, end: str):
    """Load predictions from DuckDB."""
    conn = duckdb.connect(pred_duckdb, read_only=True)
    query = """
        SELECT ts as timestamp, y_pred
        FROM predictions
        WHERE model_path = ?
          AND ts >= ?
          AND ts <= ?
        ORDER BY ts
    """
    df = conn.execute(query, [model_path, start, end]).df()
    conn.close()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


@st.cache_data
def load_ohlcv(ohlcv_duckdb: str, table: str, start: str, end: str):
    """Load OHLCV data from DuckDB."""
    conn = duckdb.connect(ohlcv_duckdb, read_only=True)
    query = f"""
        SELECT timestamp, open, high, low, close, volume
        FROM {table}
        WHERE timestamp >= ?
          AND timestamp <= ?
        ORDER BY timestamp
    """
    df = conn.execute(query, [start, end]).df()
    conn.close()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def apply_bins_for_month(df: pd.DataFrame, bin_meta, month_col='month'):
    """Apply calibration and binning to predictions."""
    p_cal, q_bin = apply_calibration_and_bins(df['y_pred'], bin_meta)
    result = df.copy()
    result['p_cal'] = p_cal
    result['q_bin'] = q_bin
    return result


def generate_signals(df: pd.DataFrame, long_bins: list[int], short_bins: list[int]):
    """Generate long/short signals based on bin assignments."""
    df = df.copy()

    # Initialize signal column
    df['signal'] = 0

    # Assign signals
    if long_bins:
        df.loc[df['q_bin'].isin(long_bins), 'signal'] = 1
    if short_bins:
        df.loc[df['q_bin'].isin(short_bins), 'signal'] = -1

    return df


def plot_strategy(df_ohlcv: pd.DataFrame, df_signals: pd.DataFrame):
    """Plot price chart with long/short entry markers."""
    # Merge signals with OHLCV
    df = df_ohlcv.merge(
        df_signals[['timestamp', 'signal', 'q_bin', 'p_cal']],
        on='timestamp',
        how='left'
    )

    # Create figure
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=('Price with Entry Signals', 'Bin Assignments')
    )

    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            increasing_line_color='green',
            decreasing_line_color='red',
        ),
        row=1, col=1
    )

    # Long signals
    long_entries = df[df['signal'] == 1]
    if not long_entries.empty:
        fig.add_trace(
            go.Scatter(
                x=long_entries['timestamp'],
                y=long_entries['low'] * 0.998,  # Slightly below low
                mode='markers',
                name='Long Entry',
                marker=dict(
                    symbol='triangle-up',
                    size=12,
                    color='green',
                    line=dict(width=1, color='darkgreen')
                ),
                hovertemplate='<b>Long Entry</b><br>Time: %{x}<br>Bin: %{customdata[0]}<br>p_cal: %{customdata[1]:.4f}<extra></extra>',
                customdata=long_entries[['q_bin', 'p_cal']].values,
            ),
            row=1, col=1
        )

    # Short signals
    short_entries = df[df['signal'] == -1]
    if not short_entries.empty:
        fig.add_trace(
            go.Scatter(
                x=short_entries['timestamp'],
                y=short_entries['high'] * 1.002,  # Slightly above high
                mode='markers',
                name='Short Entry',
                marker=dict(
                    symbol='triangle-down',
                    size=12,
                    color='red',
                    line=dict(width=1, color='darkred')
                ),
                hovertemplate='<b>Short Entry</b><br>Time: %{x}<br>Bin: %{customdata[0]}<br>p_cal: %{customdata[1]:.4f}<extra></extra>',
                customdata=short_entries[['q_bin', 'p_cal']].values,
            ),
            row=1, col=1
        )

    # Bin assignments (only where we have predictions)
    df_with_bins = df.dropna(subset=['q_bin'])
    if not df_with_bins.empty:
        fig.add_trace(
            go.Scatter(
                x=df_with_bins['timestamp'],
                y=df_with_bins['q_bin'],
                mode='markers',
                name='Bin',
                marker=dict(
                    size=4,
                    color=df_with_bins['q_bin'],
                    colorscale='RdYlGn',
                    cmin=1,
                    cmax=20,
                    colorbar=dict(title="Bin", y=0.15, len=0.3),
                ),
                hovertemplate='<b>Bin %{y}</b><br>Time: %{x}<br>p_cal: %{customdata:.4f}<extra></extra>',
                customdata=df_with_bins['p_cal'].values,
            ),
            row=2, col=1
        )

    # Update layout
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Bin (1-20)", row=2, col=1, range=[0, 21])

    fig.update_layout(
        height=800,
        showlegend=True,
        hovermode='x unified',
        xaxis_rangeslider_visible=False,
    )

    return fig


def main():
    st.title("📊 ML Calibrated Bins Strategy Analyzer")

    st.markdown("""
    Analyze trading strategies based on calibrated bin assignments.
    Configure which bins trigger long/short entries and visualize them on the price chart.
    """)

    # Sidebar configuration
    st.sidebar.header("Configuration")

    # Database paths
    pred_duckdb = st.sidebar.text_input(
        "Predictions DuckDB",
        value="/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_prediction_classifier.duckdb"
    )

    ohlcv_duckdb = st.sidebar.text_input(
        "OHLCV DuckDB",
        value="/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_ohlcv.duckdb"
    )

    ohlcv_table = st.sidebar.text_input(
        "OHLCV Table",
        value="ohlcv_btcusdt_1h"
    )

    model_path = st.sidebar.text_input(
        "Model Path",
        value="/Volumes/Extreme SSD/trading_data/cex/models/binance_btcusdt_perp_1h_original/run_20251102_231428_lgbm_y_tp_before_sl_u0.04_d0.02_24h_binary/model.txt"
    )

    # Extract run name
    run_name = Path(model_path).parent.name

    dataset = st.sidebar.text_input(
        "Dataset",
        value="binance_btcusdt_perp_1h"
    )

    bins_root = st.sidebar.text_input(
        "Bins Root",
        value="./strategies/ml_calibrated_bins/bins"
    )

    # Get available date range from predictions
    min_ts, max_ts = get_prediction_time_range(pred_duckdb, model_path)

    # Set defaults
    # Start date is always 2025-05-01 (when bin tables start)
    default_start = pd.Timestamp("2025-05-01").date()

    # End date is dynamic based on latest prediction
    if max_ts:
        default_end = max_ts.date()
        st.sidebar.info(f"Latest prediction: {default_end}")
    else:
        default_end = pd.Timestamp("2025-06-30").date()
        st.sidebar.warning("Could not detect latest prediction. Using default end date.")

    # Date range
    st.sidebar.subheader("Analysis Period")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=default_start)
    with col2:
        end_date = st.date_input("End Date", value=default_end)

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    # Strategy configuration
    st.sidebar.subheader("Strategy Configuration")

    st.sidebar.markdown("**Long Entry Bins** (select multiple)")
    long_bins = st.sidebar.multiselect(
        "Long bins",
        options=list(range(1, 21)),
        default=[19, 20],
        label_visibility="collapsed"
    )

    st.sidebar.markdown("**Short Entry Bins** (select multiple)")
    short_bins = st.sidebar.multiselect(
        "Short bins",
        options=list(range(1, 21)),
        default=[1],
        label_visibility="collapsed"
    )

    # Load data button
    if st.sidebar.button("Load Data & Run Analysis", type="primary"):
        with st.spinner("Loading data..."):
            try:
                # Load predictions
                df_pred = load_predictions(pred_duckdb, model_path, str(start), str(end))

                if df_pred.empty:
                    st.error("No predictions found for the selected period.")
                    return

                # Load OHLCV
                df_ohlcv = load_ohlcv(ohlcv_duckdb, ohlcv_table, str(start), str(end))

                if df_ohlcv.empty:
                    st.error("No OHLCV data found for the selected period.")
                    return

                # Determine months and apply bins
                df_pred['month'] = df_pred['timestamp'].dt.to_period('M').astype(str)
                months = df_pred['month'].unique()

                # Process each month
                results = []
                for month in sorted(months):
                    df_month = df_pred[df_pred['month'] == month].copy()

                    try:
                        # Load bin metadata for this month
                        bin_meta = load_bin_metadata(
                            bins_root=bins_root,
                            dataset=dataset,
                            run_name=run_name,
                            bin_month=month
                        )

                        # Apply calibration and binning
                        df_scored = apply_bins_for_month(df_month, bin_meta)
                        results.append(df_scored)

                    except FileNotFoundError:
                        st.warning(f"Bin metadata not found for {month}, skipping...")
                        continue

                if not results:
                    st.error("No bin metadata found for any month in the selected period.")
                    return

                # Combine all months
                df_all = pd.concat(results, ignore_index=True)

                # Generate signals
                df_signals = generate_signals(df_all, long_bins, short_bins)

                # Store in session state
                st.session_state['df_ohlcv'] = df_ohlcv
                st.session_state['df_signals'] = df_signals
                st.session_state['long_bins'] = long_bins
                st.session_state['short_bins'] = short_bins

                st.success(f"Loaded {len(df_signals)} predictions across {len(results)} month(s)")

            except Exception as e:
                st.error(f"Error loading data: {e}")
                import traceback
                st.code(traceback.format_exc())
                return

    # Display results if data is loaded
    if 'df_signals' in st.session_state:
        df_ohlcv = st.session_state['df_ohlcv']
        df_signals = st.session_state['df_signals']
        long_bins = st.session_state['long_bins']
        short_bins = st.session_state['short_bins']

        # Summary statistics
        st.header("📈 Strategy Summary")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_predictions = len(df_signals)
            st.metric("Total Predictions", f"{total_predictions:,}")

        with col2:
            long_signals = (df_signals['signal'] == 1).sum()
            long_pct = (long_signals / total_predictions * 100) if total_predictions > 0 else 0
            st.metric("Long Signals", f"{long_signals:,}", f"{long_pct:.1f}%")

        with col3:
            short_signals = (df_signals['signal'] == -1).sum()
            short_pct = (short_signals / total_predictions * 100) if total_predictions > 0 else 0
            st.metric("Short Signals", f"{short_signals:,}", f"{short_pct:.1f}%")

        with col4:
            neutral = (df_signals['signal'] == 0).sum()
            neutral_pct = (neutral / total_predictions * 100) if total_predictions > 0 else 0
            st.metric("Neutral", f"{neutral:,}", f"{neutral_pct:.1f}%")

        # Strategy configuration display
        st.markdown(f"""
        **Current Strategy:**
        - Long Entry: Bins {long_bins}
        - Short Entry: Bins {short_bins}
        """)

        # Price chart with signals
        st.header("📊 Price Chart with Entry Signals")
        fig = plot_strategy(df_ohlcv, df_signals)
        st.plotly_chart(fig, use_container_width=True)

        # Bin distribution
        st.header("📊 Bin Distribution (Overall)")

        col1, col2 = st.columns(2)

        with col1:
            # Histogram of bins
            bin_counts = df_signals['q_bin'].value_counts().sort_index()

            fig_hist = go.Figure()
            fig_hist.add_trace(go.Bar(
                x=bin_counts.index,
                y=bin_counts.values,
                marker_color=['green' if b in long_bins else 'red' if b in short_bins else 'gray'
                              for b in bin_counts.index],
                text=bin_counts.values,
                textposition='auto',
            ))
            fig_hist.update_layout(
                title="Prediction Count by Bin",
                xaxis_title="Bin",
                yaxis_title="Count",
                height=400,
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with col2:
            # Calibrated probability distribution
            fig_prob = go.Figure()
            fig_prob.add_trace(go.Histogram(
                x=df_signals['p_cal'],
                nbinsx=50,
                marker_color='blue',
                opacity=0.7,
            ))
            fig_prob.update_layout(
                title="Calibrated Probability Distribution",
                xaxis_title="Calibrated Probability",
                yaxis_title="Count",
                height=400,
            )
            st.plotly_chart(fig_prob, use_container_width=True)

        # Monthly breakdown
        st.header("📅 Monthly Analysis")

        # Add month column if not exists
        if 'month' not in df_signals.columns:
            df_signals['month'] = df_signals['timestamp'].dt.to_period('M').astype(str)

        months = sorted(df_signals['month'].unique())

        # Bin assignment by month
        st.subheader("Bin Assignments by Month")

        # Create stacked bar chart
        bin_data = []
        for month in months:
            df_month = df_signals[df_signals['month'] == month]
            bin_counts_month = df_month['q_bin'].value_counts().sort_index()
            for bin_num in range(1, 21):
                bin_data.append({
                    'month': month,
                    'bin': bin_num,
                    'count': bin_counts_month.get(bin_num, 0)
                })

        df_bin_monthly = pd.DataFrame(bin_data)

        fig_bins_monthly = go.Figure()

        for bin_num in range(1, 21):
            df_bin = df_bin_monthly[df_bin_monthly['bin'] == bin_num]
            color = 'green' if bin_num in long_bins else 'red' if bin_num in short_bins else 'lightgray'

            fig_bins_monthly.add_trace(go.Bar(
                name=f'Bin {bin_num}',
                x=df_bin['month'],
                y=df_bin['count'],
                marker_color=color,
                hovertemplate=f'<b>Bin {bin_num}</b><br>Month: %{{x}}<br>Count: %{{y}}<extra></extra>',
            ))

        fig_bins_monthly.update_layout(
            barmode='stack',
            title="Bin Distribution by Month (Stacked)",
            xaxis_title="Month",
            yaxis_title="Prediction Count",
            height=500,
            showlegend=False,
        )
        st.plotly_chart(fig_bins_monthly, use_container_width=True)

        # Probability distribution by month
        st.subheader("Calibrated Probability Distribution by Month")

        fig_prob_monthly = go.Figure()

        for month in months:
            df_month = df_signals[df_signals['month'] == month]

            fig_prob_monthly.add_trace(go.Histogram(
                x=df_month['p_cal'],
                name=month,
                nbinsx=30,
                opacity=0.6,
                histnorm='probability',
            ))

        fig_prob_monthly.update_layout(
            barmode='overlay',
            title="Calibrated Probability Distribution by Month",
            xaxis_title="Calibrated Probability",
            yaxis_title="Probability Density",
            height=500,
            showlegend=True,
        )
        st.plotly_chart(fig_prob_monthly, use_container_width=True)

        # Monthly statistics table
        st.subheader("Monthly Statistics")

        monthly_stats = []
        for month in months:
            df_month = df_signals[df_signals['month'] == month]
            total = len(df_month)
            long_count = (df_month['signal'] == 1).sum()
            short_count = (df_month['signal'] == -1).sum()
            neutral_count = (df_month['signal'] == 0).sum()

            monthly_stats.append({
                'Month': month,
                'Total Predictions': total,
                'Long Signals': long_count,
                'Long %': f"{(long_count/total*100):.1f}%" if total > 0 else "0%",
                'Short Signals': short_count,
                'Short %': f"{(short_count/total*100):.1f}%" if total > 0 else "0%",
                'Neutral': neutral_count,
                'Neutral %': f"{(neutral_count/total*100):.1f}%" if total > 0 else "0%",
                'Avg p_cal': f"{df_month['p_cal'].mean():.4f}",
                'Median Bin': int(df_month['q_bin'].median()),
            })

        df_monthly_stats = pd.DataFrame(monthly_stats)
        st.dataframe(df_monthly_stats, use_container_width=True)

        # Data table
        st.header("📋 Signal Details")

        # Filter options
        filter_signal = st.selectbox(
            "Filter by signal",
            options=['All', 'Long', 'Short', 'Neutral'],
            index=0
        )

        df_display = df_signals.copy()
        if filter_signal == 'Long':
            df_display = df_display[df_display['signal'] == 1]
        elif filter_signal == 'Short':
            df_display = df_display[df_display['signal'] == -1]
        elif filter_signal == 'Neutral':
            df_display = df_display[df_display['signal'] == 0]

        # Add signal label
        df_display['signal_label'] = df_display['signal'].map({1: 'LONG', -1: 'SHORT', 0: 'NEUTRAL'})

        # Display table
        st.dataframe(
            df_display[['timestamp', 'y_pred', 'p_cal', 'q_bin', 'signal_label']].sort_values('timestamp', ascending=False),
            use_container_width=True,
            height=400,
        )

        # Download data
        st.download_button(
            label="Download Signals as CSV",
            data=df_signals.to_csv(index=False),
            file_name=f"signals_{start_date}_to_{end_date}.csv",
            mime="text/csv",
        )


if __name__ == '__main__':
    main()
