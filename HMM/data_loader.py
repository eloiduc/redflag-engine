"""
data_loader.py
--------------
Fetches BTC-USD hourly OHLCV data for the last 730 days using yfinance,
then computes all technical indicators required by the trading strategy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf


# ---------------------------------------------------------------------------
# Data Fetching
# ---------------------------------------------------------------------------

def fetch_data(ticker: str = "BTC-USD", period_days: int = 730) -> pd.DataFrame:
    """
    Download hourly OHLCV data for *ticker* spanning *period_days* days.

    yfinance supports up to 730 days at the 1h interval.
    Returns a clean DataFrame with columns: Open, High, Low, Close, Volume.
    """
    df = yf.download(
        ticker,
        period=f"{period_days}d",
        interval="1h",
        progress=False,
        auto_adjust=True,
    )

    # Flatten multi-level columns produced by newer yfinance versions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)
    df.sort_index(inplace=True)
    return df


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100.0 - (100.0 / (1.0 + rs))


# ---------------------------------------------------------------------------
# ADX  (Wilder smoothing, period=14)
# ---------------------------------------------------------------------------

def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high  = df["High"]
    low   = df["Low"]
    close = df["Close"]

    # True Range
    tr = pd.concat(
        [high - low,
         (high - close.shift(1)).abs(),
         (low  - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)

    # Raw Directional Movement
    up_move   = high - high.shift(1)
    down_move = low.shift(1) - low

    pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    pos_dm = pd.Series(pos_dm, index=df.index)
    neg_dm = pd.Series(neg_dm, index=df.index)

    alpha = 1.0 / period
    atr    = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    pos_di = 100.0 * pos_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean() / (atr + 1e-10)
    neg_di = 100.0 * neg_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean() / (atr + 1e-10)

    dx  = 100.0 * (pos_di - neg_di).abs() / (pos_di + neg_di + 1e-10)
    adx = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    return adx


# ---------------------------------------------------------------------------
# Main indicator pipeline
# ---------------------------------------------------------------------------

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicators and HMM feature columns.

    HMM Features
    ------------
    Returns   : 1-period log return
    Range     : (High - Low) / Close  — normalised intrabar range
    VolVol    : rolling 20-bar coefficient-of-variation of Volume

    Strategy Signals
    ----------------
    RSI           : 14-period RSI
    Momentum      : 20-bar price change as %
    Volatility    : 20-bar rolling std of Returns × √24  (≈ daily vol %)
    VolSMA20      : 20-bar simple moving average of Volume
    ADX           : 14-period Average Directional Index
    EMA50 / EMA200: Exponential Moving Averages
    MACD          : EMA12 − EMA26
    MACD_Signal   : 9-bar EMA of MACD
    """
    out = df.copy()

    # --- HMM features ---
    out["Returns"] = np.log(out["Close"] / out["Close"].shift(1))
    out["Range"]   = (out["High"] - out["Low"]) / (out["Close"] + 1e-10)
    vol_mean       = out["Volume"].rolling(20).mean()
    vol_std        = out["Volume"].rolling(20).std()
    out["VolVol"]  = vol_std / (vol_mean + 1e-10)

    # --- Strategy indicators ---
    out["RSI"]      = _rsi(out["Close"], 14)
    out["Momentum"] = out["Close"].pct_change(20) * 100.0          # %
    out["Volatility"] = (
        out["Returns"].rolling(20).std() * np.sqrt(24) * 100.0     # daily vol %
    )
    out["VolSMA20"] = out["Volume"].rolling(20).mean()
    out["ADX"]      = _adx(out, 14)

    ema12 = out["Close"].ewm(span=12,  adjust=False).mean()
    ema26 = out["Close"].ewm(span=26,  adjust=False).mean()
    out["EMA50"]        = out["Close"].ewm(span=50,  adjust=False).mean()
    out["EMA200"]       = out["Close"].ewm(span=200, adjust=False).mean()
    out["MACD"]         = ema12 - ema26
    out["MACD_Signal"]  = out["MACD"].ewm(span=9, adjust=False).mean()

    out.dropna(inplace=True)
    return out
