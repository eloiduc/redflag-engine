"""
app.py
------
Streamlit dashboard for the Regime-Based BTC/USD Trading System.

Layout
  ┌─────────────────────────────────────────────────────────────┐
  │ Signal card │ Regime card │ BTC Price │ Confirmations badge  │
  ├─────────────────────────────────────────────────────────────┤
  │ Total Return │ Alpha │ Win Rate │ Max DD │ Final Capital      │
  ├─────────────────────────────────────────────────────────────┤
  │        Interactive Plotly candlestick + regime BG           │
  │        Equity sub-chart                                     │
  │        Volume sub-chart                                     │
  ├─────────────────────────────────────────────────────────────┤
  │ Regime Distribution  │  HMM State Summary                   │
  ├─────────────────────────────────────────────────────────────┤
  │ Full trade log                                              │
  └─────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from backtester import (
    CONFIRM_NEEDED,
    INITIAL_CAPITAL,
    REGIME_BEAR,
    REGIME_BULL,
    SIGNAL_NAMES,
    _confirmations,
    run_full_pipeline,
)

# ───────────────────────────────────────────────────────────────
# Page config
# ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="HMM Regime Trader · BTC/USD",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ───────────────────────────────────────────────────────────────
# Custom CSS
# ───────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* metric cards */
    [data-testid="stMetric"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 14px 18px;
    }
    [data-testid="stMetricLabel"]  { color: #8b949e !important; font-size: 0.78rem; }
    [data-testid="stMetricValue"]  { font-size: 1.35rem !important; }
    [data-testid="stMetricDelta"]  { font-size: 0.80rem !important; }
    /* title */
    h1 { letter-spacing: -1px; }
    /* divider */
    hr { border-color: #30363d; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ───────────────────────────────────────────────────────────────
# Regime colour palette
# ───────────────────────────────────────────────────────────────

REGIME_COLORS: dict[str, str] = {
    "Bull Run":   "rgba(0,  200, 100, 0.18)",
    "Bullish":    "rgba(0,  160,  60, 0.09)",
    "Neutral":    "rgba(120,120,120, 0.06)",
    "Bearish":    "rgba(220,120,  0, 0.10)",
    "Bear/Crash": "rgba(220,  50, 50, 0.20)",
}

REGIME_LINE: dict[str, str] = {
    "Bull Run":   "#00c864",
    "Bullish":    "#26a69a",
    "Neutral":    "#888888",
    "Bearish":    "#ff9800",
    "Bear/Crash": "#ef5350",
}


# ───────────────────────────────────────────────────────────────
# Data loading (cached)
# ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=3_600, show_spinner=False)
def load_pipeline():
    return run_full_pipeline()


# ───────────────────────────────────────────────────────────────
# Chart builder
# ───────────────────────────────────────────────────────────────

def build_chart(
    df: pd.DataFrame,
    equity_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    show_ema: bool,
    show_trades: bool,
    lookback_days: int,
) -> go.Figure:
    """
    Build a 3-panel interactive Plotly chart:
      Row 1 – Candlestick with regime background + EMAs + trade markers
      Row 2 – Equity curve
      Row 3 – Volume bars
    """
    # --- Slice to lookback window ---
    cutoff = df.index[-1] - pd.Timedelta(days=lookback_days)
    view_df = df[df.index >= cutoff]
    view_eq  = equity_df[equity_df.index >= cutoff]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=[0.60, 0.22, 0.18],
        subplot_titles=("BTC/USD Price & Market Regimes", "Strategy Equity ($)", "Volume"),
    )

    # ── Regime backgrounds (group consecutive identical regimes) ──────────
    regime_change_id = view_df["Regime"].ne(view_df["Regime"].shift()).cumsum()
    for _, seg in view_df.groupby(regime_change_id, sort=False):
        regime = seg["Regime"].iloc[0]
        x0, x1 = seg.index[0], seg.index[-1]
        fill   = REGIME_COLORS.get(regime, "rgba(100,100,100,0.05)")
        for r in (1,):
            fig.add_vrect(
                x0=x0, x1=x1,
                fillcolor=fill, opacity=1,
                layer="below", line_width=0,
                row=r, col=1,
            )

    # ── Candlestick ───────────────────────────────────────────────────────
    fig.add_trace(
        go.Candlestick(
            x=view_df.index,
            open=view_df["Open"], high=view_df["High"],
            low=view_df["Low"],  close=view_df["Close"],
            name="BTC/USD",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            increasing_fillcolor="#26a69a",
            decreasing_fillcolor="#ef5350",
            line=dict(width=1),
        ),
        row=1, col=1,
    )

    # ── EMAs ─────────────────────────────────────────────────────────────
    if show_ema:
        fig.add_trace(
            go.Scatter(
                x=view_df.index, y=view_df["EMA50"],
                name="EMA 50", line=dict(color="#f0b429", width=1.2),
                opacity=0.85,
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=view_df.index, y=view_df["EMA200"],
                name="EMA 200", line=dict(color="#4db8ff", width=1.2),
                opacity=0.85,
            ),
            row=1, col=1,
        )

    # ── Trade markers ────────────────────────────────────────────────────
    if show_trades and len(trades_df) > 0:
        view_trades = trades_df[trades_df["entry_time"] >= cutoff]
        if len(view_trades):
            fig.add_trace(
                go.Scatter(
                    x=view_trades["entry_time"],
                    y=view_trades["entry_price"],
                    mode="markers",
                    name="Entry",
                    marker=dict(
                        symbol="triangle-up",
                        size=12,
                        color="#00c864",
                        line=dict(color="white", width=1),
                    ),
                ),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=view_trades["exit_time"],
                    y=view_trades["exit_price"],
                    mode="markers",
                    name="Exit",
                    marker=dict(
                        symbol="triangle-down",
                        size=12,
                        color="#ef5350",
                        line=dict(color="white", width=1),
                    ),
                ),
                row=1, col=1,
            )

    # ── Equity curve ─────────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=view_eq.index,
            y=view_eq["equity"],
            name="Equity",
            line=dict(color="#00d4ff", width=2),
            fill="tozeroy",
            fillcolor="rgba(0,212,255,0.07)",
        ),
        row=2, col=1,
    )
    # Buy-and-hold reference
    bh_start  = float(view_df["Close"].iloc[0])
    bh_series = view_df["Close"] / bh_start * INITIAL_CAPITAL
    fig.add_trace(
        go.Scatter(
            x=view_df.index,
            y=bh_series,
            name="Buy & Hold",
            line=dict(color="#ff9800", width=1.5, dash="dot"),
            opacity=0.7,
        ),
        row=2, col=1,
    )

    # ── Volume ────────────────────────────────────────────────────────────
    vol_colors = [
        "#26a69a" if c >= o else "#ef5350"
        for c, o in zip(view_df["Close"], view_df["Open"])
    ]
    fig.add_trace(
        go.Bar(
            x=view_df.index,
            y=view_df["Volume"],
            name="Volume",
            marker_color=vol_colors,
            opacity=0.65,
        ),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=view_df.index,
            y=view_df["VolSMA20"],
            name="Vol SMA20",
            line=dict(color="#ff9800", width=1),
            opacity=0.8,
        ),
        row=3, col=1,
    )

    # ── Layout ────────────────────────────────────────────────────────────
    fig.update_layout(
        template="plotly_dark",
        height=860,
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(
            orientation="h",
            x=0, y=1.03,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11),
        ),
        font=dict(family="JetBrains Mono, Courier New, monospace", size=11),
    )
    fig.update_yaxes(gridcolor="#1c2128", gridwidth=0.5)
    fig.update_xaxes(gridcolor="#1c2128", gridwidth=0.5, showspikes=True, spikecolor="#888")

    return fig


# ───────────────────────────────────────────────────────────────
# Regime distribution pie
# ───────────────────────────────────────────────────────────────

def build_regime_pie(df: pd.DataFrame) -> go.Figure:
    counts = df["Regime"].value_counts()
    colors = [REGIME_LINE.get(r, "#888888") for r in counts.index]

    fig = go.Figure(go.Pie(
        labels=counts.index,
        values=counts.values,
        hole=0.45,
        marker=dict(colors=colors, line=dict(color="#0d1117", width=2)),
        textinfo="label+percent",
        textfont=dict(size=12),
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        height=300,
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
    )
    return fig


# ───────────────────────────────────────────────────────────────
# Main app
# ───────────────────────────────────────────────────────────────

def main() -> None:

    # ── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Controls")
        lookback_days = st.slider(
            "Chart lookback (days)", min_value=30, max_value=730, value=180, step=30
        )
        show_ema    = st.toggle("Show EMAs (50 / 200)", value=True)
        show_trades = st.toggle("Show Trade Markers",   value=True)
        st.divider()
        st.markdown("### Strategy Parameters")
        st.markdown(f"- **Leverage**: 2.5×")
        st.markdown(f"- **Cooldown**: 48 h after exit")
        st.markdown(f"- **Min confirmations**: {CONFIRM_NEEDED}/8")
        st.markdown(f"- **HMM States**: 7")
        st.divider()
        force_reload = st.button("🔄 Reload Data", use_container_width=True)

    if force_reload:
        st.cache_data.clear()

    # ── Load pipeline ─────────────────────────────────────────────────────
    with st.spinner("Loading data and running backtest … (first run ~30 s)"):
        df, equity_df, trades_df, metrics, engine = load_pipeline()

    # ── Header ────────────────────────────────────────────────────────────
    st.markdown(
        "<h1 style='font-size:1.9rem;'>📈 HMM Regime-Based Trading · BTC/USD</h1>",
        unsafe_allow_html=True,
    )
    st.caption(
        "7-State GaussianHMM · 8-Confirmation Voting System · 2.5× Leverage · 48 h Cooldown"
    )

    # ── Current signal row ───────────────────────────────────────────────
    last_row    = df.iloc[-1]
    cur_regime  = last_row["Regime"]
    cur_price   = float(last_row["Close"])
    conf_cnt, conf_list = _confirmations(last_row)
    signal      = "LONG" if cur_regime == REGIME_BULL and conf_cnt >= CONFIRM_NEEDED else "CASH"
    sig_color   = "#00c864" if signal == "LONG" else "#ef5350"
    reg_color   = REGIME_LINE.get(cur_regime, "#888")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"<div style='background:#161b22;border:1px solid {sig_color};"
            f"border-radius:10px;padding:16px;text-align:center'>"
            f"<p style='color:#8b949e;font-size:.78rem;margin:0'>CURRENT SIGNAL</p>"
            f"<p style='color:{sig_color};font-size:2rem;font-weight:700;margin:4px 0'>{signal}</p>"
            f"<p style='color:#8b949e;font-size:.78rem;margin:0'>{conf_cnt}/8 confirmations</p>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"<div style='background:#161b22;border:1px solid {reg_color};"
            f"border-radius:10px;padding:16px;text-align:center'>"
            f"<p style='color:#8b949e;font-size:.78rem;margin:0'>DETECTED REGIME</p>"
            f"<p style='color:{reg_color};font-size:1.5rem;font-weight:700;margin:4px 0'>{cur_regime}</p>"
            f"<p style='color:#8b949e;font-size:.78rem;margin:0'>HMM State {int(last_row['HMM_State'])}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.metric("BTC Price", f"${cur_price:,.2f}")
        data_span = (df.index[-1] - df.index[0]).days
        st.caption(f"Data: {data_span} days · {len(df):,} bars")
    with c4:
        st.metric("Bull / Bear State", f"{engine.bull_state} / {engine.bear_state}")
        st.caption(f"Cooldown: 48 h · Leverage: 2.5×")

    st.divider()

    # ── Confirmation signals grid ─────────────────────────────────────────
    with st.expander("🔍 Current Confirmation Signals", expanded=False):
        cols = st.columns(4)
        for i, (name, val) in enumerate(zip(SIGNAL_NAMES, conf_list)):
            icon  = "✅" if val else "❌"
            color = "#00c864" if val else "#ef5350"
            with cols[i % 4]:
                st.markdown(
                    f"<div style='background:#161b22;border-radius:6px;"
                    f"padding:8px 12px;margin-bottom:6px;border-left:3px solid {color}'>"
                    f"<span style='color:{color}'>{icon}</span> "
                    f"<span style='font-size:.88rem'>{name}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # ── Performance metrics ───────────────────────────────────────────────
    st.subheader("Performance Metrics")
    m1, m2, m3, m4, m5 = st.columns(5)

    tr_delta = f"B&H: {metrics['bh_return']:.1f}%"
    with m1:
        st.metric("Total Return", f"{metrics['total_return']:.1f}%", delta=tr_delta,
                  delta_color="normal")
    with m2:
        sign = "+" if metrics["alpha"] > 0 else ""
        st.metric("Alpha vs B&H", f"{sign}{metrics['alpha']:.1f}%",
                  delta_color="normal")
    with m3:
        st.metric("Win Rate", f"{metrics['win_rate']:.1f}%",
                  delta=f"{metrics['n_trades']} trades")
    with m4:
        st.metric("Max Drawdown", f"{metrics['max_drawdown']:.1f}%",
                  delta_color="inverse")
    with m5:
        st.metric(
            "Final Capital",
            f"${metrics['final_capital']:,.0f}",
            delta=f"${metrics['final_capital'] - INITIAL_CAPITAL:+,.0f}",
        )

    st.divider()

    # ── Main chart ────────────────────────────────────────────────────────
    st.subheader("Interactive Chart")
    fig = build_chart(df, equity_df, trades_df, show_ema, show_trades, lookback_days)
    st.plotly_chart(fig, use_container_width=True)

    # ── Regime legend ─────────────────────────────────────────────────────
    cols_leg = st.columns(5)
    regime_order = ["Bull Run", "Bullish", "Neutral", "Bearish", "Bear/Crash"]
    for col, r in zip(cols_leg, regime_order):
        c = REGIME_LINE.get(r, "#888")
        col.markdown(
            f"<div style='background:{REGIME_COLORS.get(r,'rgba(128,128,128,.1)')};"
            f"border-left:4px solid {c};border-radius:4px;"
            f"padding:4px 10px;text-align:center;font-size:.82rem;color:{c}'>"
            f"■ {r}</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Analytics row ─────────────────────────────────────────────────────
    col_pie, col_state = st.columns([1, 1])

    with col_pie:
        st.subheader("Regime Distribution")
        st.plotly_chart(build_regime_pie(df), use_container_width=True)

    with col_state:
        st.subheader("HMM State Summary")
        state_tbl = engine.state_summary().reset_index()
        state_tbl.insert(0, "Active", state_tbl["State"].apply(
            lambda s: "🐂 Bull" if s == engine.bull_state
            else ("🐻 Bear" if s == engine.bear_state else "—")
        ))
        st.dataframe(state_tbl, use_container_width=True, hide_index=True)

        if metrics["n_trades"] > 0:
            st.markdown("**Win / Loss stats**")
            wl1, wl2, wl3 = st.columns(3)
            wl1.metric("Avg Win",  f"${metrics['avg_win']:+,.0f}")
            wl2.metric("Avg Loss", f"${metrics['avg_loss']:+,.0f}")
            ratio = abs(metrics["avg_win"] / metrics["avg_loss"]) if metrics["avg_loss"] != 0 else float("inf")
            wl3.metric("Reward : Risk", f"{ratio:.2f}")

    st.divider()

    # ── Trade log ─────────────────────────────────────────────────────────
    if len(trades_df) > 0:
        st.subheader(f"Trade Log · {len(trades_df)} trades")
        display_df = trades_df.copy()
        display_df["entry_time"] = display_df["entry_time"].dt.strftime("%Y-%m-%d %H:%M")
        display_df["exit_time"]  = display_df["exit_time"].dt.strftime("%Y-%m-%d %H:%M")
        display_df["hold_hours"] = display_df["hold_hours"].map("{:.1f} h".format)
        display_df["pnl_pct"]    = display_df["pnl_pct"].map("{:+.2f}%".format)
        display_df["pnl_dollar"] = display_df["pnl_dollar"].map("${:+,.2f}".format)
        display_df["entry_price"] = display_df["entry_price"].map("${:,.2f}".format)
        display_df["exit_price"]  = display_df["exit_price"].map("${:,.2f}".format)
        display_df["capital_after"] = display_df["capital_after"].map("${:,.2f}".format)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("No trades were generated for this period. The strategy requires Bull Run regime + 7/8 confirmations simultaneously.", icon="ℹ️")

    # ── Footer ────────────────────────────────────────────────────────────
    st.divider()
    st.caption(
        "Regime-Based HMM Trading System · "
        "7-State GaussianHMM trained on Returns / Range / Volume Volatility · "
        "Past performance is not indicative of future results."
    )


if __name__ == "__main__":
    main()
