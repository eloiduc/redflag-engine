"""
backtester.py
-------------
Core HMM engine + backtest simulation.

HMM
  - 7-component GaussianHMM trained on [Returns, Range, VolVol]
  - Automatically identifies Bull-Run state (highest mean return)
    and Bear/Crash state (lowest mean return)

Strategy
  - Entry  : HMM == Bull Run AND ≥ 7/8 technical confirmations
  - Exit   : HMM flips to Bear/Crash state
  - Cooldown: 48-hour hard cooldown after every exit
  - Leverage: 2.5x on all PnL

Backtest
  - Starting capital : $10,000
  - Every trade is logged with entry/exit time, prices, PnL, reason
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

from data_loader import compute_indicators, fetch_data

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_STATES        = 7
INITIAL_CAPITAL = 10_000.0
LEVERAGE        = 2.5
COOLDOWN_HOURS  = 48
CONFIRM_NEEDED  = 7       # out of 8 signals


# ---------------------------------------------------------------------------
# Regime labels
# ---------------------------------------------------------------------------

REGIME_BULL  = "Bull Run"
REGIME_BEAR  = "Bear/Crash"
REGIME_BULL2 = "Bullish"
REGIME_BEAR2 = "Bearish"
REGIME_NEUT  = "Neutral"


# ---------------------------------------------------------------------------
# HMM Engine
# ---------------------------------------------------------------------------

@dataclass
class HMMEngine:
    n_states: int = N_STATES
    model:    Optional[hmm.GaussianHMM] = field(default=None, repr=False)
    scaler:   Optional[StandardScaler]  = field(default=None, repr=False)
    bull_state: int = -1
    bear_state: int = -1
    regime_labels: dict[int, str] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame) -> "HMMEngine":
        """Train the HMM on Returns, Range, VolVol features."""
        X = df[["Returns", "Range", "VolVol"]].values

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=500,
            tol=1e-5,
            random_state=42,
        )
        self.model.fit(X_scaled)

        self._identify_states()
        return self

    def _identify_states(self) -> None:
        """Map each HMM state to a human-readable regime label."""
        # mean Returns for each state (unscale only the first feature)
        scaled_means = self.model.means_[:, 0]
        returns_std  = self.scaler.scale_[0]
        returns_mean = self.scaler.mean_[0]
        raw_means    = scaled_means * returns_std + returns_mean

        self.bull_state = int(np.argmax(raw_means))
        self.bear_state = int(np.argmin(raw_means))

        self.regime_labels = {}
        for i in range(self.n_states):
            if i == self.bull_state:
                self.regime_labels[i] = REGIME_BULL
            elif i == self.bear_state:
                self.regime_labels[i] = REGIME_BEAR
            elif raw_means[i] > 0:
                self.regime_labels[i] = REGIME_BULL2
            elif raw_means[i] < -1e-4:
                self.regime_labels[i] = REGIME_BEAR2
            else:
                self.regime_labels[i] = REGIME_NEUT

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return predicted state sequence for each row in *df*."""
        X = df[["Returns", "Range", "VolVol"]].values
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def state_summary(self) -> pd.DataFrame:
        """Return a summary table of each state's mean features."""
        means_scaled = self.model.means_
        means_raw    = self.scaler.inverse_transform(means_scaled)
        rows = []
        for i in range(self.n_states):
            rows.append({
                "State":       i,
                "Label":       self.regime_labels[i],
                "Mean Return": f"{means_raw[i, 0] * 100:.4f}%",
                "Mean Range":  f"{means_raw[i, 1] * 100:.4f}%",
                "Mean VolVol": f"{means_raw[i, 2]:.4f}",
            })
        return pd.DataFrame(rows).set_index("State")


# ---------------------------------------------------------------------------
# Confirmation signals
# ---------------------------------------------------------------------------

SIGNAL_NAMES = [
    "RSI < 90",
    "Momentum > 1%",
    "Volatility < 6%",
    "Volume > SMA20",
    "ADX > 25",
    "Price > EMA50",
    "Price > EMA200",
    "MACD > Signal",
]


def _confirmations(row: pd.Series) -> tuple[int, list[bool]]:
    """
    Evaluate the 8 confirmation signals for a single bar.
    Returns (count_of_True, list_of_bool).
    """
    signals = [
        row["RSI"]      < 90,
        row["Momentum"] > 1.0,
        row["Volatility"] < 6.0,
        row["Volume"]   > row["VolSMA20"],
        row["ADX"]      > 25.0,
        row["Close"]    > row["EMA50"],
        row["Close"]    > row["EMA200"],
        row["MACD"]     > row["MACD_Signal"],
    ]
    return sum(signals), signals


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------

def run_backtest(
    df: pd.DataFrame,
    engine: HMMEngine,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulate the strategy bar-by-bar.

    Parameters
    ----------
    df     : fully featured DataFrame (from compute_indicators)
    engine : fitted HMMEngine

    Returns
    -------
    equity_df : hourly equity curve with regime / position columns
    trades_df : log of every completed trade
    """
    hidden_states = engine.predict(df)
    df = df.copy()
    df["HMM_State"] = hidden_states
    df["Regime"]    = [engine.regime_labels[s] for s in hidden_states]

    capital      = INITIAL_CAPITAL
    position     = None          # None or dict with entry info
    cooldown_end = pd.Timestamp.min.tz_localize(df.index.tz)

    equity_rows: list[dict] = []
    trade_rows:  list[dict] = []

    for ts, row in df.iterrows():
        regime        = row["Regime"]
        conf_cnt, _   = _confirmations(row)
        in_cooldown   = ts < cooldown_end

        # ── EXIT ─────────────────────────────────────────────────────────────
        if position is not None and regime == REGIME_BEAR:
            exit_price   = float(row["Close"])
            raw_ret      = (exit_price - position["entry_price"]) / position["entry_price"]
            lev_ret      = raw_ret * LEVERAGE
            pnl_dollar   = position["capital"] * lev_ret
            capital     += pnl_dollar
            capital      = max(capital, 0.0)

            trade_rows.append({
                "entry_time":    position["entry_time"],
                "exit_time":     ts,
                "entry_price":   position["entry_price"],
                "exit_price":    exit_price,
                "hold_hours":    (ts - position["entry_time"]).total_seconds() / 3600,
                "pnl_pct":       lev_ret * 100.0,
                "pnl_dollar":    pnl_dollar,
                "exit_reason":   "Bear/Crash Regime",
                "capital_after": capital,
            })

            cooldown_end = ts + pd.Timedelta(hours=COOLDOWN_HOURS)
            position     = None

        # ── ENTRY ────────────────────────────────────────────────────────────
        if position is None and not in_cooldown:
            if regime == REGIME_BULL and conf_cnt >= CONFIRM_NEEDED:
                position = {
                    "entry_time":  ts,
                    "entry_price": float(row["Close"]),
                    "capital":     capital,
                }

        # ── EQUITY SNAPSHOT ──────────────────────────────────────────────────
        if position is not None:
            raw_ret    = (float(row["Close"]) - position["entry_price"]) / position["entry_price"]
            cur_equity = position["capital"] * (1.0 + raw_ret * LEVERAGE)
        else:
            cur_equity = capital

        equity_rows.append({
            "time":          ts,
            "equity":        cur_equity,
            "regime":        regime,
            "hmm_state":     int(row["HMM_State"]),
            "in_position":   position is not None,
            "confirmations": conf_cnt,
            "in_cooldown":   in_cooldown,
        })

    # ── CLOSE ANY OPEN POSITION AT END ───────────────────────────────────────
    if position is not None:
        last_row     = df.iloc[-1]
        last_ts      = df.index[-1]
        exit_price   = float(last_row["Close"])
        raw_ret      = (exit_price - position["entry_price"]) / position["entry_price"]
        lev_ret      = raw_ret * LEVERAGE
        pnl_dollar   = position["capital"] * lev_ret
        capital     += pnl_dollar
        capital      = max(capital, 0.0)

        trade_rows.append({
            "entry_time":    position["entry_time"],
            "exit_time":     last_ts,
            "entry_price":   position["entry_price"],
            "exit_price":    exit_price,
            "hold_hours":    (last_ts - position["entry_time"]).total_seconds() / 3600,
            "pnl_pct":       lev_ret * 100.0,
            "pnl_dollar":    pnl_dollar,
            "exit_reason":   "End of Data",
            "capital_after": capital,
        })

        equity_rows[-1]["equity"]      = capital
        equity_rows[-1]["in_position"] = False

    equity_df = pd.DataFrame(equity_rows).set_index("time")
    trades_df = pd.DataFrame(trade_rows) if trade_rows else pd.DataFrame()
    return equity_df, trades_df


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    equity_df:      pd.DataFrame,
    trades_df:      pd.DataFrame,
    df:             pd.DataFrame,
    initial_capital: float = INITIAL_CAPITAL,
) -> dict:
    """Compute Total Return, Alpha, Win Rate, and Max Drawdown."""
    final_equity = equity_df["equity"].iloc[-1]
    total_return = (final_equity - initial_capital) / initial_capital * 100.0

    bh_return = (df["Close"].iloc[-1] - df["Close"].iloc[0]) / df["Close"].iloc[0] * 100.0
    alpha     = total_return - bh_return

    if len(trades_df) > 0:
        wins     = (trades_df["pnl_dollar"] > 0).sum()
        win_rate = wins / len(trades_df) * 100.0
        avg_win  = trades_df.loc[trades_df["pnl_dollar"] > 0, "pnl_dollar"].mean()
        avg_loss = trades_df.loc[trades_df["pnl_dollar"] <= 0, "pnl_dollar"].mean()
    else:
        win_rate = 0.0
        avg_win  = 0.0
        avg_loss = 0.0

    rolling_max  = equity_df["equity"].cummax()
    drawdown     = (equity_df["equity"] - rolling_max) / rolling_max * 100.0
    max_drawdown = float(drawdown.min())

    return {
        "total_return":   total_return,
        "bh_return":      bh_return,
        "alpha":          alpha,
        "win_rate":       win_rate,
        "max_drawdown":   max_drawdown,
        "final_capital":  final_equity,
        "n_trades":       len(trades_df),
        "avg_win":        avg_win  if not np.isnan(avg_win)  else 0.0,
        "avg_loss":       avg_loss if not np.isnan(avg_loss) else 0.0,
    }


# ---------------------------------------------------------------------------
# Full pipeline (called by app.py via @st.cache_data)
# ---------------------------------------------------------------------------

def run_full_pipeline() -> tuple:
    """
    Orchestrates the full workflow:
      1. Fetch + compute indicators
      2. Train HMM
      3. Backtest
      4. Compute metrics

    Returns
    -------
    df, equity_df, trades_df, metrics, engine
    """
    print("  [1/4] Fetching BTC-USD hourly data …")
    df_raw = fetch_data()

    print("  [2/4] Computing technical indicators …")
    df = compute_indicators(df_raw)

    print("  [3/4] Training 7-state GaussianHMM …")
    engine = HMMEngine(n_states=N_STATES)
    engine.fit(df)
    print(f"        Bull Run state → {engine.bull_state}  |  "
          f"Bear/Crash state → {engine.bear_state}")
    print(engine.state_summary().to_string())

    # Attach regime columns to df
    states     = engine.predict(df)
    df["HMM_State"] = states
    df["Regime"]    = [engine.regime_labels[s] for s in states]

    print("  [4/4] Running backtest …")
    equity_df, trades_df = run_backtest(df, engine)

    metrics = compute_metrics(equity_df, trades_df, df)
    print(f"        Total Return: {metrics['total_return']:.1f}%  |  "
          f"Alpha: {metrics['alpha']:.1f}%  |  "
          f"Win Rate: {metrics['win_rate']:.1f}%  |  "
          f"Max DD: {metrics['max_drawdown']:.1f}%")

    return df, equity_df, trades_df, metrics, engine


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df, equity_df, trades_df, metrics, engine = run_full_pipeline()
    print("\n── Trades ──")
    if len(trades_df):
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 160)
        print(trades_df.to_string(index=False))
    else:
        print("  No trades generated.")
    print("\n── Metrics ──")
    for k, v in metrics.items():
        print(f"  {k:<18}: {v:.2f}")
