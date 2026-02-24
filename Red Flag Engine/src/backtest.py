"""backtest.py — Post-earnings return computation.

For each report that has an entry in earnings_dates.json, fetches historical
price data via yfinance and computes post-earnings returns at 1, 5, and 20
trading-day horizons.

yfinance is imported lazily inside the functions that need it so that the
module can be imported without it installed (the functions return None/empty
gracefully when yfinance is absent).

No LLM calls — fully deterministic.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

# Filename pattern: {COMPANY}_{NOW}_vs_{PREV}.md
_STEM_RE = re.compile(r"^([A-Z0-9]+)_(.+)_vs_(.+)$", re.IGNORECASE)

# Calendar days to fetch — enough to cover 20+ trading days
_FETCH_DAYS = 45


# ---------------------------------------------------------------------------
# Pydantic model
# ---------------------------------------------------------------------------

class PostEarningsReturns(BaseModel):
    """Post-earnings price returns for a single report / earnings call."""

    model_config = ConfigDict(extra="forbid")

    ticker:    str
    call_date: str            # ISO 8601 date of the earnings call
    ret_1d:    Optional[float] = None   # 1-trading-day return
    ret_5d:    Optional[float] = None   # 5-trading-day return
    ret_20d:   Optional[float] = None   # 20-trading-day return


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_earnings_dates(path: Path) -> dict[str, str]:
    """Load earnings_dates.json; return empty dict on any error."""
    if not path.exists():
        logger.debug("earnings_dates.json not found at %s", path)
        return {}
    try:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        # Drop the _comment key if present
        return {k: v for k, v in data.items() if not k.startswith("_")}
    except Exception as exc:
        logger.warning("Failed to load earnings_dates.json: %s", exc)
        return {}


def _ret(close_series, t: int, baseline: float) -> Optional[float]:
    """Compute return at index *t* relative to *baseline*.  Returns None if out of range."""
    if t < len(close_series) and baseline != 0:
        return round((close_series.iloc[t] - baseline) / baseline, 4)
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_post_earnings_returns(
    report_filename:     str,
    earnings_dates_path: Path,
) -> Optional[PostEarningsReturns]:
    """Compute post-earnings price returns for one report.

    Looks up the earnings call date in earnings_dates.json using the report
    filename stem (without ``.md``), then fetches price history via yfinance
    and computes 1d / 5d / 20d returns.

    Args:
        report_filename:     E.g. ``"BA_2025Q4_vs_2025Q3.md"``.
        earnings_dates_path: Path to earnings_dates.json.

    Returns:
        :class:`PostEarningsReturns` on success, or ``None`` on any error
        (missing date entry, network failure, import error, etc.).
    """
    # ── Lazy import of yfinance ───────────────────────────────────────────
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not installed — backtest unavailable. Run: pip install yfinance")
        return None

    # ── Look up earnings date ─────────────────────────────────────────────
    stem = re.sub(r"\.md$", "", report_filename, flags=re.IGNORECASE)
    dates = _load_earnings_dates(earnings_dates_path)
    call_date_str = dates.get(stem)
    if not call_date_str:
        logger.debug("No earnings date for report stem '%s'", stem)
        return None

    # ── Parse ticker from stem (first token before first underscore) ──────
    m = _STEM_RE.match(stem)
    if not m:
        logger.debug("Could not parse ticker from stem '%s'", stem)
        return None
    ticker = m.group(1).upper()

    # ── Fetch price history ───────────────────────────────────────────────
    try:
        from datetime import date, timedelta
        call_date = date.fromisoformat(call_date_str)
        end_date  = call_date + timedelta(days=_FETCH_DAYS)

        t = yf.Ticker(ticker)
        hist = t.history(
            start=call_date.isoformat(),
            end=end_date.isoformat(),
            auto_adjust=True,
        )
    except Exception as exc:
        logger.warning("yfinance fetch failed for %s (%s): %s", ticker, call_date_str, exc)
        return None

    if hist.empty or len(hist) < 2:
        logger.warning("Insufficient price data for %s starting %s", ticker, call_date_str)
        return None

    close = hist["Close"]
    baseline = close.iloc[0]
    if baseline == 0:
        return None

    result = PostEarningsReturns(
        ticker    = ticker,
        call_date = call_date_str,
        ret_1d    = _ret(close, 1,  baseline),
        ret_5d    = _ret(close, 5,  baseline),
        ret_20d   = _ret(close, 20, baseline),
    )

    logger.info(
        "Backtest %s @ %s: 1d=%s  5d=%s  20d=%s",
        ticker, call_date_str,
        f"{result.ret_1d:+.1%}" if result.ret_1d is not None else "N/A",
        f"{result.ret_5d:+.1%}" if result.ret_5d is not None else "N/A",
        f"{result.ret_20d:+.1%}" if result.ret_20d is not None else "N/A",
    )
    return result


def load_backtest_summary(
    outputs_dir:         Path,
    earnings_dates_path: Path,
) -> "pd.DataFrame":
    """Build an aggregate backtest DataFrame from all reports in outputs_dir.

    Args:
        outputs_dir:         Directory containing generated .md reports.
        earnings_dates_path: Path to earnings_dates.json.

    Returns:
        pandas DataFrame with columns: Report, Ticker, Date, 1d%, 5d%, 20d%.
        Returns an empty DataFrame when no data is available (no yfinance,
        no date entries, or network failure).
    """
    try:
        import pandas as pd
    except ImportError:
        logger.warning("pandas not available — cannot build backtest summary")
        import types
        return types.SimpleNamespace(empty=True)  # type: ignore[return-value]

    rows: list[dict] = []

    for md_path in sorted(outputs_dir.glob("*.md")):
        result = compute_post_earnings_returns(md_path.name, earnings_dates_path)
        if result is None:
            continue

        def _pct(r: Optional[float]) -> str:
            return f"{r:+.1%}" if r is not None else "—"

        rows.append({
            "Report":  md_path.name,
            "Ticker":  result.ticker,
            "Date":    result.call_date,
            "1d%":     _pct(result.ret_1d),
            "5d%":     _pct(result.ret_5d),
            "20d%":    _pct(result.ret_20d),
        })

    logger.info("Backtest summary: %d entries", len(rows))
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["Report", "Ticker", "Date", "1d%", "5d%", "20d%"]
    )
