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


def generate_signals_per_month(
    df: pd.DataFrame,
    monthly_bins: dict[str, dict[str, list[int]]]
):
    """Generate long/short signals with per-month bin configurations.

    Args:
        df: DataFrame with 'timestamp', 'q_bin' columns
        monthly_bins: Dict mapping month -> {'long': [bins], 'short': [bins]}
                      e.g., {'2025-05': {'long': [19, 20], 'short': [1]}, ...}

    Returns:
        DataFrame with 'signal' column added
    """
    df = df.copy()

    # Ensure month column exists
    if 'month' not in df.columns:
        df['month'] = df['timestamp'].dt.to_period('M').astype(str)

    # Initialize signal column
    df['signal'] = 0

    # Process each month
    for month, bins_config in monthly_bins.items():
        month_mask = df['month'] == month
        long_bins = bins_config.get('long', [])
        short_bins = bins_config.get('short', [])

        if long_bins:
            df.loc[month_mask & df['q_bin'].isin(long_bins), 'signal'] = 1
        if short_bins:
            df.loc[month_mask & df['q_bin'].isin(short_bins), 'signal'] = -1

    return df


def simulate_pnl(
    df_signals: pd.DataFrame,
    df_ohlcv: pd.DataFrame,
    initial_balance: float = 10000.0,
    leverage: float = 1.0,
    long_tp_pct: float = 0.04,  # +4% for long TP
    long_sl_pct: float = 0.02,  # -2% for long SL
    short_tp_pct: float = 0.02,  # -2% for short TP (price drop)
    short_sl_pct: float = 0.04,  # +4% for short SL (price rise)
    max_bars: int = 24,
    position_fraction: float = 1/24,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulate PnL for the strategy with independent trades.

    Rules:
    - Entry at T+1 open when signal triggers at T
    - TP/SL checked using high/low from T+1 to T+24
    - Exit at T+24 close if neither TP nor SL triggers
    - Position size: 1/24 of balance for first entry, same size for subsequent
    - Leverage multiplies the exposure (and thus PnL)

    Args:
        df_signals: DataFrame with 'timestamp' and 'signal' columns
        df_ohlcv: DataFrame with OHLCV data
        initial_balance: Starting balance
        leverage: Leverage multiplier (1.0 = no leverage, 10.0 = 10x)
        long_tp_pct: Take profit % for longs (0.04 = 4%)
        long_sl_pct: Stop loss % for longs (0.02 = 2%)
        short_tp_pct: Take profit % for shorts (price drop)
        short_sl_pct: Stop loss % for shorts (price rise)
        max_bars: Maximum bars to hold position
        position_fraction: Fraction of balance for position sizing

    Returns:
        trades_df: DataFrame with all trades
        equity_df: DataFrame with equity curve
    """
    # Create timestamp index for OHLCV for fast lookup
    df_ohlcv = df_ohlcv.copy().set_index('timestamp').sort_index()

    # Filter signals (only non-zero)
    signals = df_signals[df_signals['signal'] != 0].copy()
    signals = signals.sort_values('timestamp')

    trades = []
    balance = initial_balance
    last_position_size = None

    for _, row in signals.iterrows():
        signal_time = row['timestamp']
        direction = row['signal']  # 1 for long, -1 for short

        # Find T+1 bar for entry
        try:
            future_times = df_ohlcv.index[df_ohlcv.index > signal_time]
            if len(future_times) < 1:
                continue  # Not enough future data
            entry_time = future_times[0]
            entry_price = df_ohlcv.loc[entry_time, 'open']
        except (KeyError, IndexError):
            continue

        # Determine position size
        if last_position_size is None:
            position_size = balance * position_fraction
        else:
            position_size = last_position_size

        last_position_size = position_size

        # Calculate TP/SL prices
        if direction == 1:  # Long
            tp_price = entry_price * (1 + long_tp_pct)
            sl_price = entry_price * (1 - long_sl_pct)
        else:  # Short
            tp_price = entry_price * (1 - short_tp_pct)
            sl_price = entry_price * (1 + short_sl_pct)

        # Scan forward bars to find exit
        exit_time = None
        exit_price = None
        exit_reason = None

        future_bars = df_ohlcv.index[df_ohlcv.index >= entry_time][:max_bars]

        for i, bar_time in enumerate(future_bars):
            bar = df_ohlcv.loc[bar_time]

            if direction == 1:  # Long
                # Check SL first (more conservative)
                if bar['low'] <= sl_price:
                    exit_time = bar_time
                    exit_price = sl_price
                    exit_reason = 'SL'
                    break
                # Check TP
                if bar['high'] >= tp_price:
                    exit_time = bar_time
                    exit_price = tp_price
                    exit_reason = 'TP'
                    break
            else:  # Short
                # Check SL first (price rising)
                if bar['high'] >= sl_price:
                    exit_time = bar_time
                    exit_price = sl_price
                    exit_reason = 'SL'
                    break
                # Check TP (price dropping)
                if bar['low'] <= tp_price:
                    exit_time = bar_time
                    exit_price = tp_price
                    exit_reason = 'TP'
                    break

        # If no TP/SL, exit at T+24 close
        if exit_time is None:
            if len(future_bars) >= max_bars:
                exit_time = future_bars[max_bars - 1]
                exit_price = df_ohlcv.loc[exit_time, 'close']
                exit_reason = 'TIMEOUT'
            else:
                # Not enough data to complete trade
                continue

        # Calculate PnL
        if direction == 1:  # Long
            pnl_pct = (exit_price - entry_price) / entry_price
        else:  # Short
            pnl_pct = (entry_price - exit_price) / entry_price

        # Apply leverage: PnL = margin × leverage × price_change_pct
        pnl_usd = position_size * leverage * pnl_pct
        balance += pnl_usd

        trades.append({
            'signal_time': signal_time,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'direction': 'LONG' if direction == 1 else 'SHORT',
            'entry_price': entry_price,
            'exit_price': exit_price,
            'tp_price': tp_price,
            'sl_price': sl_price,
            'exit_reason': exit_reason,
            'position_size': position_size,
            'leverage': leverage,
            'exposure': position_size * leverage,
            'pnl_pct': pnl_pct,
            'pnl_usd': pnl_usd,
            'balance_after': balance,
            'q_bin': row['q_bin'],
            'p_cal': row['p_cal'],
        })

    trades_df = pd.DataFrame(trades)

    # Create equity curve
    if not trades_df.empty:
        equity_data = [{'timestamp': trades_df['signal_time'].min() - pd.Timedelta(hours=1),
                        'equity': initial_balance}]
        for _, trade in trades_df.iterrows():
            equity_data.append({
                'timestamp': trade['exit_time'],
                'equity': trade['balance_after']
            })
        equity_df = pd.DataFrame(equity_data)
    else:
        equity_df = pd.DataFrame({'timestamp': [], 'equity': []})

    return trades_df, equity_df


def calculate_trade_statistics(trades_df: pd.DataFrame, initial_balance: float = 10000.0) -> dict:
    """Calculate trading statistics from trades DataFrame."""
    if trades_df.empty:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'total_return_pct': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'max_drawdown_pct': 0,
            'sharpe_ratio': 0,
            'long_trades': 0,
            'short_trades': 0,
            'long_win_rate': 0,
            'short_win_rate': 0,
            'tp_exits': 0,
            'sl_exits': 0,
            'timeout_exits': 0,
        }

    total_trades = len(trades_df)
    winning_trades = (trades_df['pnl_usd'] > 0).sum()
    losing_trades = (trades_df['pnl_usd'] < 0).sum()
    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    total_pnl = trades_df['pnl_usd'].sum()
    final_balance = trades_df['balance_after'].iloc[-1]
    total_return_pct = (final_balance - initial_balance) / initial_balance * 100

    # Average win/loss
    wins = trades_df[trades_df['pnl_usd'] > 0]['pnl_usd']
    losses = trades_df[trades_df['pnl_usd'] < 0]['pnl_usd']
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0

    # Profit factor
    gross_profit = wins.sum() if len(wins) > 0 else 0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Max drawdown
    equity_curve = trades_df['balance_after'].values
    peak = np.maximum.accumulate(np.concatenate([[initial_balance], equity_curve]))
    drawdown = (peak[1:] - equity_curve) / peak[1:]
    max_drawdown_pct = drawdown.max() * 100 if len(drawdown) > 0 else 0

    # Sharpe ratio (annualized, assuming hourly trades)
    returns = trades_df['pnl_pct'].values
    if len(returns) > 1 and returns.std() > 0:
        # Rough annualization: ~8760 hours per year
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(8760 / 24)  # per-trade basis
    else:
        sharpe_ratio = 0

    # By direction
    long_trades = trades_df[trades_df['direction'] == 'LONG']
    short_trades = trades_df[trades_df['direction'] == 'SHORT']
    long_win_rate = (long_trades['pnl_usd'] > 0).mean() if len(long_trades) > 0 else 0
    short_win_rate = (short_trades['pnl_usd'] > 0).mean() if len(short_trades) > 0 else 0

    # Exit reasons
    tp_exits = (trades_df['exit_reason'] == 'TP').sum()
    sl_exits = (trades_df['exit_reason'] == 'SL').sum()
    timeout_exits = (trades_df['exit_reason'] == 'TIMEOUT').sum()

    return {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'total_return_pct': total_return_pct,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'max_drawdown_pct': max_drawdown_pct,
        'sharpe_ratio': sharpe_ratio,
        'long_trades': len(long_trades),
        'short_trades': len(short_trades),
        'long_win_rate': long_win_rate,
        'short_win_rate': short_win_rate,
        'tp_exits': tp_exits,
        'sl_exits': sl_exits,
        'timeout_exits': timeout_exits,
    }


def plot_equity_curve(
    equity_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    df_ohlcv: pd.DataFrame,
    initial_balance: float = 10000.0
):
    """Plot equity curve with buy-and-hold comparison."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=('Equity Curve vs Buy & Hold', 'Trade PnL')
    )

    bh_equity_values = []

    # Calculate buy-and-hold equity curve
    if not equity_df.empty and not df_ohlcv.empty:
        # Get the time range from equity curve
        start_time = equity_df['timestamp'].min()
        end_time = equity_df['timestamp'].max()

        # Filter OHLCV to the same range
        df_bh = df_ohlcv[
            (df_ohlcv['timestamp'] >= start_time) &
            (df_ohlcv['timestamp'] <= end_time)
        ].copy()

        if not df_bh.empty:
            # Calculate buy-and-hold returns (using close prices)
            start_price = df_bh['close'].iloc[0]
            df_bh['bh_equity'] = initial_balance * (df_bh['close'] / start_price)
            bh_equity_values = df_bh['bh_equity'].values.tolist()

            # Add buy-and-hold curve
            fig.add_trace(
                go.Scatter(
                    x=df_bh['timestamp'],
                    y=df_bh['bh_equity'],
                    mode='lines',
                    name='Buy & Hold',
                    line=dict(color='orange', width=2, dash='dot'),
                    opacity=0.8,
                ),
                row=1, col=1
            )

    # Strategy equity curve
    fig.add_trace(
        go.Scatter(
            x=equity_df['timestamp'],
            y=equity_df['equity'],
            mode='lines',
            name='Strategy',
            line=dict(color='blue', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 100, 255, 0.1)',
        ),
        row=1, col=1
    )

    # Calculate y-axis range to center curves with padding
    all_values = list(equity_df['equity'].values) + bh_equity_values + [initial_balance]

    y_min = min(all_values)
    y_max = max(all_values)
    y_range = y_max - y_min
    padding = y_range * 0.15  # 15% padding on each side

    # Initial balance line
    fig.add_hline(
        y=initial_balance,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Initial: ${initial_balance:,.0f}",
        row=1, col=1
    )

    # Trade PnL bars
    if not trades_df.empty:
        colors = ['green' if pnl > 0 else 'red' for pnl in trades_df['pnl_usd']]
        fig.add_trace(
            go.Bar(
                x=trades_df['exit_time'],
                y=trades_df['pnl_usd'],
                name='Trade PnL',
                marker_color=colors,
                hovertemplate='<b>%{customdata[0]}</b><br>PnL: $%{y:.2f}<br>Exit: %{customdata[1]}<extra></extra>',
                customdata=trades_df[['direction', 'exit_reason']].values,
            ),
            row=2, col=1
        )

    fig.update_layout(
        height=600,
        showlegend=True,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
    )
    # Set y-axis range with padding to center the curves
    fig.update_yaxes(
        title_text="Equity ($)",
        range=[y_min - padding, y_max + padding],
        row=1, col=1
    )
    fig.update_yaxes(title_text="PnL ($)", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)

    return fig


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
    end = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

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

    st.sidebar.markdown("---")

    # Buttons
    col_btn1, col_btn2 = st.sidebar.columns(2)

    with col_btn1:
        if st.button("🔄 Refresh Data", help="Clear cache and reload from database"):
            st.cache_data.clear()
            st.rerun()

    with col_btn2:
        load_button = st.button("Load & Analyze", type="primary")

    # Load data button
    if load_button:
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
                st.session_state['df_all'] = df_all  # Store raw binned data for simulation
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

        # Create tabs
        tab_analysis, tab_simulation = st.tabs(["📊 Analysis", "💰 PnL Simulation"])

        with tab_analysis:
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

        with tab_simulation:
            # PnL Simulation Section
            st.header("💰 PnL Simulation")

            st.markdown("""
            Simulate trading performance with the following rules:
            - **Position sizing**: 1/24 of balance for first entry, same size for subsequent entries
            - **Entry**: T+1 open price when signal triggers at T
            - **Exit**: TP/SL triggers (using high/low), or T+24 close if neither triggers
            - **Long**: TP = +4%, SL = -2%
            - **Short**: TP = -2% (price drop), SL = +4% (price rise)
            """)

            # Get available months from data
            df_all = st.session_state['df_all']
            if 'month' not in df_all.columns:
                df_all['month'] = df_all['timestamp'].dt.to_period('M').astype(str)
            available_months = sorted(df_all['month'].unique())

            # Bin configuration section
            st.subheader("🎯 Entry Bin Configuration")

            # Default bins (apply to all months unless overridden)
            st.markdown("**Default Bins** (applied to all months)")
            col_def1, col_def2 = st.columns(2)
            with col_def1:
                default_long_bins = st.multiselect(
                    "Default Long Entry Bins",
                    options=list(range(1, 21)),
                    default=long_bins,
                    key='default_long_bins',
                    help="Default bins for LONG entries. Can be overridden per month below."
                )
            with col_def2:
                default_short_bins = st.multiselect(
                    "Default Short Entry Bins",
                    options=list(range(1, 21)),
                    default=short_bins,
                    key='default_short_bins',
                    help="Default bins for SHORT entries. Can be overridden per month below."
                )

            # Per-month configuration
            st.markdown("---")
            st.markdown("**Per-Month Configuration** (override defaults for specific months)")

            # Initialize monthly bins config
            monthly_bins_config = {}

            # Create expanders for each month
            for month in available_months:
                with st.expander(f"📅 {month}", expanded=False):
                    col_m1, col_m2, col_m3 = st.columns([2, 2, 1])

                    with col_m1:
                        month_long = st.multiselect(
                            f"Long Bins",
                            options=list(range(1, 21)),
                            default=default_long_bins,
                            key=f'long_{month}',
                        )
                    with col_m2:
                        month_short = st.multiselect(
                            f"Short Bins",
                            options=list(range(1, 21)),
                            default=default_short_bins,
                            key=f'short_{month}',
                        )
                    with col_m3:
                        # Show if this differs from default
                        is_custom = (month_long != default_long_bins) or (month_short != default_short_bins)
                        if is_custom:
                            st.markdown("⚡ **Custom**")
                        else:
                            st.markdown("📋 Default")

                    monthly_bins_config[month] = {
                        'long': month_long,
                        'short': month_short
                    }

            # Summary of configuration
            st.markdown("---")
            custom_months = [m for m, cfg in monthly_bins_config.items()
                           if cfg['long'] != default_long_bins or cfg['short'] != default_short_bins]
            if custom_months:
                st.info(f"**Custom config for:** {', '.join(custom_months)}")
            else:
                st.info(f"**All months using default:** Long = {default_long_bins}, Short = {default_short_bins}")

            # Simulation parameters
            st.markdown("---")
            col_sim1, col_sim2, col_sim3 = st.columns(3)
            with col_sim1:
                initial_balance = st.number_input(
                    "Initial Balance ($)",
                    min_value=100.0,
                    max_value=1000000.0,
                    value=10000.0,
                    step=1000.0
                )
            with col_sim2:
                leverage = st.number_input(
                    "Leverage",
                    min_value=1.0,
                    max_value=20.0,
                    value=1.0,
                    step=1.0,
                    help="Leverage multiplier (1x-20x). Higher leverage = higher risk/reward."
                )
            with col_sim3:
                run_simulation = st.button("🚀 Run Simulation", type="primary")

            if run_simulation or 'trades_df' in st.session_state:
                if run_simulation:
                    with st.spinner("Running PnL simulation..."):
                        # Generate signals with per-month bin configuration
                        df_sim_signals = generate_signals_per_month(df_all, monthly_bins_config)

                        trades_df, equity_df = simulate_pnl(
                            df_signals=df_sim_signals,
                            df_ohlcv=df_ohlcv,
                            initial_balance=initial_balance,
                            leverage=leverage,
                            long_tp_pct=0.04,
                            long_sl_pct=0.02,
                            short_tp_pct=0.02,
                            short_sl_pct=0.04,
                            max_bars=24,
                            position_fraction=1/24,
                        )
                        st.session_state['trades_df'] = trades_df
                        st.session_state['equity_df'] = equity_df
                        st.session_state['sim_initial_balance'] = initial_balance
                        st.session_state['sim_leverage'] = leverage
                        st.session_state['sim_monthly_bins'] = monthly_bins_config.copy()
                        st.session_state['sim_default_long'] = default_long_bins
                        st.session_state['sim_default_short'] = default_short_bins

                trades_df = st.session_state['trades_df']
                equity_df = st.session_state['equity_df']
                sim_initial_balance = st.session_state.get('sim_initial_balance', 10000.0)
                sim_leverage = st.session_state.get('sim_leverage', 1.0)
                sim_monthly_bins = st.session_state.get('sim_monthly_bins', {})
                sim_default_long = st.session_state.get('sim_default_long', [])
                sim_default_short = st.session_state.get('sim_default_short', [])

                if trades_df.empty:
                    st.warning("No trades executed. Check if signals have enough future OHLCV data.")
                else:
                    # Show configuration summary
                    custom_months = [m for m, cfg in sim_monthly_bins.items()
                                    if cfg['long'] != sim_default_long or cfg['short'] != sim_default_short]
                    if custom_months:
                        st.success(f"**Simulation ran with:** Default Long={sim_default_long}, Short={sim_default_short}, Leverage={sim_leverage:.0f}x | Custom months: {', '.join(custom_months)}")
                    else:
                        st.success(f"**Simulation ran with:** Long={sim_default_long}, Short={sim_default_short}, Leverage={sim_leverage:.0f}x (all months)")

                    # Calculate statistics
                    stats = calculate_trade_statistics(trades_df, sim_initial_balance)

                    # Display key metrics
                    st.subheader("📊 Performance Metrics")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        final_balance = trades_df['balance_after'].iloc[-1]
                        delta_color = "normal" if final_balance >= sim_initial_balance else "inverse"
                        st.metric(
                            "Final Balance",
                            f"${final_balance:,.2f}",
                            f"{stats['total_return_pct']:+.2f}%",
                            delta_color=delta_color
                        )
                    with col2:
                        st.metric("Total Trades", stats['total_trades'])
                    with col3:
                        st.metric("Win Rate", f"{stats['win_rate']*100:.1f}%")
                    with col4:
                        st.metric("Profit Factor", f"{stats['profit_factor']:.2f}" if stats['profit_factor'] != float('inf') else "∞")

                    col5, col6, col7, col8 = st.columns(4)
                    with col5:
                        st.metric("Total PnL", f"${stats['total_pnl']:,.2f}")
                    with col6:
                        st.metric("Max Drawdown", f"{stats['max_drawdown_pct']:.2f}%")
                    with col7:
                        st.metric("Avg Win", f"${stats['avg_win']:.2f}")
                    with col8:
                        st.metric("Avg Loss", f"${stats['avg_loss']:.2f}")

                    # Direction breakdown
                    st.subheader("📈 By Direction")
                    col_dir1, col_dir2, col_dir3, col_dir4 = st.columns(4)
                    with col_dir1:
                        st.metric("Long Trades", stats['long_trades'])
                    with col_dir2:
                        st.metric("Long Win Rate", f"{stats['long_win_rate']*100:.1f}%")
                    with col_dir3:
                        st.metric("Short Trades", stats['short_trades'])
                    with col_dir4:
                        st.metric("Short Win Rate", f"{stats['short_win_rate']*100:.1f}%")

                    # Exit reasons
                    st.subheader("🚪 Exit Breakdown")
                    col_exit1, col_exit2, col_exit3 = st.columns(3)
                    with col_exit1:
                        tp_pct = stats['tp_exits'] / stats['total_trades'] * 100 if stats['total_trades'] > 0 else 0
                        st.metric("TP Exits", f"{stats['tp_exits']}", f"{tp_pct:.1f}%")
                    with col_exit2:
                        sl_pct = stats['sl_exits'] / stats['total_trades'] * 100 if stats['total_trades'] > 0 else 0
                        st.metric("SL Exits", f"{stats['sl_exits']}", f"{sl_pct:.1f}%")
                    with col_exit3:
                        timeout_pct = stats['timeout_exits'] / stats['total_trades'] * 100 if stats['total_trades'] > 0 else 0
                        st.metric("Timeout Exits", f"{stats['timeout_exits']}", f"{timeout_pct:.1f}%")

                    # Equity curve
                    st.subheader("📈 Equity Curve")
                    fig_equity = plot_equity_curve(equity_df, trades_df, df_ohlcv, sim_initial_balance)
                    st.plotly_chart(fig_equity, use_container_width=True)

                    # Monthly PnL breakdown
                    st.subheader("📅 Monthly PnL")
                    trades_df['exit_month'] = pd.to_datetime(trades_df['exit_time']).dt.to_period('M').astype(str)
                    monthly_pnl = trades_df.groupby('exit_month').agg({
                        'pnl_usd': 'sum',
                        'signal_time': 'count',
                        'exit_reason': lambda x: (x == 'TP').sum()
                    }).rename(columns={'signal_time': 'trades', 'exit_reason': 'tp_count'})
                    monthly_pnl['win_rate'] = trades_df.groupby('exit_month').apply(
                        lambda x: (x['pnl_usd'] > 0).mean()
                    )

                    fig_monthly = go.Figure()
                    colors = ['green' if pnl > 0 else 'red' for pnl in monthly_pnl['pnl_usd']]
                    fig_monthly.add_trace(go.Bar(
                        x=monthly_pnl.index,
                        y=monthly_pnl['pnl_usd'],
                        marker_color=colors,
                        text=[f"${pnl:,.0f}" for pnl in monthly_pnl['pnl_usd']],
                        textposition='auto',
                        hovertemplate='<b>%{x}</b><br>PnL: $%{y:,.2f}<br>Trades: %{customdata[0]}<br>Win Rate: %{customdata[1]:.1%}<extra></extra>',
                        customdata=monthly_pnl[['trades', 'win_rate']].values,
                    ))
                    fig_monthly.update_layout(
                        title="Monthly PnL",
                        xaxis_title="Month",
                        yaxis_title="PnL ($)",
                        height=400,
                    )
                    st.plotly_chart(fig_monthly, use_container_width=True)

                    # Monthly statistics with bin config and long/short breakdown
                    st.subheader("📊 Monthly Statistics with Bin Configuration")

                    monthly_stats_data = []
                    for month in sorted(monthly_pnl.index):
                        month_cfg = sim_monthly_bins.get(month, {'long': sim_default_long, 'short': sim_default_short})
                        is_custom = month_cfg['long'] != sim_default_long or month_cfg['short'] != sim_default_short

                        # Filter trades for this month
                        month_trades = trades_df[trades_df['exit_month'] == month]
                        long_trades = month_trades[month_trades['direction'] == 'LONG']
                        short_trades = month_trades[month_trades['direction'] == 'SHORT']

                        # Calculate long stats
                        long_count = len(long_trades)
                        long_pnl = long_trades['pnl_usd'].sum() if long_count > 0 else 0
                        long_win_rate = (long_trades['pnl_usd'] > 0).mean() if long_count > 0 else 0

                        # Calculate short stats
                        short_count = len(short_trades)
                        short_pnl = short_trades['pnl_usd'].sum() if short_count > 0 else 0
                        short_win_rate = (short_trades['pnl_usd'] > 0).mean() if short_count > 0 else 0

                        # Total stats
                        total_pnl = monthly_pnl.loc[month, 'pnl_usd']
                        total_win_rate = monthly_pnl.loc[month, 'win_rate']

                        monthly_stats_data.append({
                            'Month': month,
                            'Long Bins': str(month_cfg['long']) if month_cfg['long'] else '-',
                            'Short Bins': str(month_cfg['short']) if month_cfg['short'] else '-',
                            '⚡': '⚡' if is_custom else '',
                            'L#': long_count,
                            'L PnL': f"${long_pnl:+,.0f}" if long_count > 0 else '-',
                            'L Win%': f"{long_win_rate*100:.0f}%" if long_count > 0 else '-',
                            'S#': short_count,
                            'S PnL': f"${short_pnl:+,.0f}" if short_count > 0 else '-',
                            'S Win%': f"{short_win_rate*100:.0f}%" if short_count > 0 else '-',
                            'Total PnL': f"${total_pnl:+,.0f}",
                            'Win%': f"{total_win_rate*100:.0f}%",
                        })

                    df_monthly_stats = pd.DataFrame(monthly_stats_data)
                    st.dataframe(df_monthly_stats, use_container_width=True, hide_index=True)

                    # Trade log
                    st.subheader("📋 Trade Log")

                    # Filter options for trades
                    trade_filter = st.selectbox(
                        "Filter trades by",
                        options=['All', 'Winning', 'Losing', 'Long', 'Short', 'TP Exit', 'SL Exit', 'Timeout'],
                        index=0,
                        key='trade_filter'
                    )

                    df_trades_display = trades_df.copy()
                    if trade_filter == 'Winning':
                        df_trades_display = df_trades_display[df_trades_display['pnl_usd'] > 0]
                    elif trade_filter == 'Losing':
                        df_trades_display = df_trades_display[df_trades_display['pnl_usd'] < 0]
                    elif trade_filter == 'Long':
                        df_trades_display = df_trades_display[df_trades_display['direction'] == 'LONG']
                    elif trade_filter == 'Short':
                        df_trades_display = df_trades_display[df_trades_display['direction'] == 'SHORT']
                    elif trade_filter == 'TP Exit':
                        df_trades_display = df_trades_display[df_trades_display['exit_reason'] == 'TP']
                    elif trade_filter == 'SL Exit':
                        df_trades_display = df_trades_display[df_trades_display['exit_reason'] == 'SL']
                    elif trade_filter == 'Timeout':
                        df_trades_display = df_trades_display[df_trades_display['exit_reason'] == 'TIMEOUT']

                    # Format for display
                    display_cols = ['signal_time', 'direction', 'leverage', 'exposure', 'entry_price', 'exit_price',
                                   'exit_reason', 'pnl_pct', 'pnl_usd', 'balance_after', 'q_bin']
                    df_trades_display = df_trades_display[display_cols].copy()
                    df_trades_display['leverage'] = df_trades_display['leverage'].apply(lambda x: f"{x:.0f}x")
                    df_trades_display['exposure'] = df_trades_display['exposure'].apply(lambda x: f"${x:,.2f}")
                    df_trades_display['pnl_pct'] = df_trades_display['pnl_pct'].apply(lambda x: f"{x*100:+.2f}%")
                    df_trades_display['pnl_usd'] = df_trades_display['pnl_usd'].apply(lambda x: f"${x:+,.2f}")
                    df_trades_display['balance_after'] = df_trades_display['balance_after'].apply(lambda x: f"${x:,.2f}")
                    df_trades_display['entry_price'] = df_trades_display['entry_price'].apply(lambda x: f"${x:,.2f}")
                    df_trades_display['exit_price'] = df_trades_display['exit_price'].apply(lambda x: f"${x:,.2f}")

                    st.dataframe(
                        df_trades_display.sort_values('signal_time', ascending=False),
                        use_container_width=True,
                        height=400,
                    )

                    # Download trades
                    st.download_button(
                        label="Download Trades as CSV",
                        data=trades_df.to_csv(index=False),
                        file_name=f"trades_{start_date}_to_{end_date}.csv",
                        mime="text/csv",
                    )


if __name__ == '__main__':
    main()
