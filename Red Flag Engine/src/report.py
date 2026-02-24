from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel, ConfigDict

from diff import Change, ChangeType, AbandonedMetric

if TYPE_CHECKING:
    from hedge_score import HedgeDelta
    from peer_contagion import PeerSignal
    from backtest import PostEarningsReturns
    from prediction_markets import PredictionMarket, MarketClaimCrossRef

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ReportStats — pipeline coverage metadata passed in from main.py
# ---------------------------------------------------------------------------

class ReportStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_chunks_now:  int = 0
    n_chunks_prev: int = 0
    n_claims_now:  int = 0
    n_claims_prev: int = 0
    n_matched:     int = 0   # strict + soft matches (change_type != new)
    n_new:         int = 0
    n_soft:        int = 0   # soft-matched changes


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TOP_SUMMARY = 7
_TOP_TABLE   = 10


_SEVERITY_LABELS: dict[int, str] = {
    5: "Critical",
    4: "High",
    3: "Medium",
    2: "Low",
    1: "Informational",
}

_CHANGE_BADGE: dict[ChangeType, str] = {
    ChangeType.new:       "🆕 NEW",
    ChangeType.worsened:  "🔴 WORSENED",
    ChangeType.improved:  "🟢 IMPROVED",
    ChangeType.unchanged: "⚪ UNCHANGED",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _severity_label(score: int) -> str:
    return _SEVERITY_LABELS.get(score, str(score))


def _escape_pipe(text: str) -> str:
    """Escape pipe characters so they don't break Markdown table cells."""
    return text.replace("|", "\\|")


def _render_header(
    company:     str,
    now_period:  str,
    prev_period: str,
    changes:     list[Change],
    stats:       ReportStats,
) -> str:
    ts            = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    high_critical = sum(1 for c in changes if c.severity >= 4)
    return "\n".join([
        f"# Red Flag Report: {company}",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| Company | **{company}** |",
        f"| Current quarter | `{now_period}` |",
        f"| Prior quarter | `{prev_period}` |",
        f"| Generated | {ts} |",
        f"| Total changes detected | {len(changes)} |",
        f"| High / Critical | **{high_critical}** |",
        "",
        "> ⚠️ **Disclaimer:** This report is a triage aid only. It is NOT a trading signal, "
        "investment recommendation, or financial advice. It may contain false positives or "
        "miss subtle language. All findings must be independently verified by a qualified analyst.",
        "",
        f"> 📊 **Coverage:** {stats.n_chunks_now} chunks analysed · "
        f"{stats.n_claims_now} claims extracted (current) · "
        f"{stats.n_claims_prev} claims extracted (prior) · "
        f"{stats.n_matched} matched · "
        f"{stats.n_new} new · "
        f"{stats.n_soft} soft-matched",
    ])


def _render_executive_summary(changes: list[Change]) -> str:
    top   = changes[:_TOP_SUMMARY]
    lines = ["## Executive Summary", ""]
    if not top:
        lines.append("_No material changes detected._")
        return "\n".join(lines)
    for c in top:
        badge = _CHANGE_BADGE.get(c.change_type, c.change_type.value.upper())
        sev   = _severity_label(c.severity)
        lines.append(f"- {badge} **[{sev}]** {c.summary}")
    return "\n".join(lines)


def _render_red_flags_table(changes: list[Change]) -> str:
    top = changes[:_TOP_TABLE]
    lines = [
        "## Red Flags",
        "",
        "| # | Category | Change | Sev | Evidence (Now) | Chunk (Now) | Evidence (Prev) | Chunk (Prev) |",
        "|---|----------|--------|-----|----------------|-------------|-----------------|--------------|",
    ]
    if not top:
        lines.append("| — | — | — | — | _No changes_ | — | — | — |")
        return "\n".join(lines)

    used_analyst = False
    used_soft    = False

    for i, c in enumerate(top, start=1):
        cat   = c.category.value.replace("_", " ").title()
        badge = _CHANGE_BADGE.get(c.change_type, c.change_type.value.upper())
        sev   = _severity_label(c.severity)

        # ── Analyst marker ────────────────────────────────────────────
        analyst_tag = ""
        if c.now_speaker_role == "analyst":
            analyst_tag = " ⁽ᴬ⁾"
            used_analyst = True

        # ── Soft-match marker ─────────────────────────────────────────
        soft_tag = ""
        if c.match_quality == "soft":
            soft_tag = " ⁽ˢ⁾"
            used_soft = True

        ev_now  = _escape_pipe(c.now_evidence) + analyst_tag + soft_tag
        ck_now  = f"`{c.now_chunk_id}`"
        ev_prev = _escape_pipe(c.prev_evidence) if c.prev_evidence else "—"
        ck_prev = f"`{c.prev_chunk_id}`" if c.prev_chunk_id else "—"

        lines.append(
            f"| {i} | {cat} | {badge} | {sev} | {ev_now} | {ck_now} | {ev_prev} | {ck_prev} |"
        )

    # ── Footnotes ─────────────────────────────────────────────────────────
    if used_analyst or used_soft:
        lines.append("")
    if used_analyst:
        lines.append(
            "> ⁽ᴬ⁾ Claim sourced from analyst question, not management statement. "
            "Treat with caution."
        )
    if used_soft:
        lines.append(
            "> ⁽ˢ⁾ Matched via relaxed similarity (soft match) — verify manually against "
            "the original transcript."
        )

    return "\n".join(lines)



def _render_limitations() -> str:
    return "\n".join([
        "## Limitations",
        "",
        "- **Triage tool only.** This engine surfaces potential narrative shifts for analyst "
        "review. It is not predictive and must not be used as a basis for trading decisions.",
        "- **False positives / negatives.** LLM extraction at temperature=0 is conservative "
        "but not infallible. Subtle hedging, irony, or boilerplate language may be "
        "misclassified.",
        "- **Transcript quality dependency.** Poor-quality PDFs, missing Q&A sections, or "
        "partial transcripts will reduce recall.",
        "- **Evidence is bounded by chunk context.** Cross-paragraph nuance may be missed "
        "if a claim spans a chunk boundary.",
        "- **No investment advice.** No financial advice is provided or implied. The authors "
        "accept no liability for decisions made based on this output.",
        "- **Not a substitute for primary source review.** Always read the original "
        "transcript before acting on any finding in this report.",
    ])


def _render_methodology() -> str:
    return "\n".join([
        "## Methodology",
        "",
        "Transcripts are split into ~3,500-character chunks at paragraph boundaries. "
        "Each chunk is labelled with an inferred section (guidance, liquidity, demand, etc.) "
        "via keyword regex and a speaker role (management / analyst / operator) via regex. "
        "Claude extracts at most 6 claims per chunk using a strict zero-temperature prompt "
        "that requires a verbatim evidence quote (≤ 25 words) for every claim; claims without "
        "valid evidence are discarded. Quarter-over-quarter change detection uses a two-pass "
        "RapidFuzz strategy: strict (`token_set_ratio` ≥ 72 on full claim text) then soft "
        "(same category, first 60 chars, ≥ 60). Severity is assigned by a deterministic "
        "heuristic based on change type, category risk, polarity, and confidence; "
        "low-confidence claims are capped at severity 3. Supplementary signals — hedging "
        "intensity, abandoned metrics, peer contagion, and backtest context — are computed "
        "deterministically with no additional LLM calls.",
    ])


def _render_abandoned_metrics(abandoned: "list[AbandonedMetric]") -> str:
    """Render the Abandoned Metrics section; returns '' if list is empty."""
    if not abandoned:
        return ""
    lines = [
        "## Abandoned Metrics",
        "",
        "The following categories were discussed in the prior quarter but appear absent "
        "from the current transcript (≥ 2 prior claims, zero fuzzy matches now).",
        "",
        "| Category | Prior Quarter Statement | Evidence | Chunk | Confidence |",
        "|----------|------------------------|----------|-------|------------|",
    ]
    for m in abandoned:
        cat = m.category.value.replace("_", " ").title()
        lines.append(
            f"| {cat} | {_escape_pipe(m.representative_claim)} "
            f"| {_escape_pipe(m.evidence)} | `{m.chunk_id}` | {m.confidence.value} |"
        )
    return "\n".join(lines)


def _render_hedging_intensity(deltas: "list[HedgeDelta]") -> str:
    """Render the Hedging Intensity section; returns '' if list is empty."""
    if not deltas:
        return ""
    lines = [
        "## Hedging Intensity",
        "",
        "Hedge word density (Tier 1: may/might/could/uncertain…; "
        "Tier 2: expect/anticipate/believe…) per 100 words, by section. "
        "⚠️ flags sections where current-quarter hedging increased by > 3 percentage points.",
        "",
        "| Section | Now (/100w) | Prev (/100w) | Chg | Flag |",
        "|---------|------------|-------------|-----|------|",
    ]
    for d in deltas:
        delta_str = f"+{d.delta:.1f}" if d.delta >= 0 else f"{d.delta:.1f}"
        flag_str  = "FLAG" if d.flag else ""
        lines.append(
            f"| {d.section} | {d.now_score:.1f} | {d.prev_score:.1f} | {delta_str} | {flag_str} |"
        )
    return "\n".join(lines)


def _render_peer_signals(signals: "list[PeerSignal]") -> str:
    """Render the Peer & Supplier Signals section; returns '' if list is empty."""
    if not signals:
        return ""
    lines = [
        "## Peer & Supplier Signals",
        "",
        "Red flags surfaced from related companies' most recent reports. "
        "These may indicate sector-level or supply-chain stress relevant to this company.",
        "",
        "| Source | Rel | Category | Evidence | Polarity | Sev | Report |",
        "|--------|-----|----------|----------|----------|-----|--------|",
    ]
    for s in signals:
        lines.append(
            f"| {s.source_company} | {s.relationship} | {s.category} "
            f"| {_escape_pipe(s.evidence)} | {s.polarity} | {s.sev} | `{s.report_filename}` |"
        )
    return "\n".join(lines)


def _render_prediction_markets(
    markets:   "list[PredictionMarket]",
    crossrefs: "list[MarketClaimCrossRef]",
) -> str:
    """Render the Prediction Market Context section; returns '' if both lists are empty."""
    if not markets and not crossrefs:
        return ""

    from datetime import date as _date
    today = _date.today().isoformat()

    lines = [
        "## Prediction Market Context",
        "",
        f"Active markets sourced from Polymarket / Kalshi as of {today}. "
        "Minimum volume: $5,000. Only markets with a strong probability signal "
        "(Yes < 35% or Yes > 65%) are cross-referenced with management claims.",
    ]

    # ── Active Markets table ──────────────────────────────────────────────
    if markets:
        lines += [
            "",
            "### Active Markets",
            "",
            "| Platform | Market | Yes % | Volume (USD) | Expires |",
            "|----------|--------|-------|--------------|---------|",
        ]
        for m in markets:
            prob_str    = f"{m.yes_probability:.0%}"
            vol_str     = f"${m.volume_usd:,.0f}"
            expires_str = m.expires or "—"
            q_display   = _escape_pipe(m.question)
            if m.url:
                q_display = f"[{q_display}]({m.url})"
            lines.append(
                f"| {m.platform} | {q_display} | {prob_str} | {vol_str} | {expires_str} |"
            )

    # ── Cross-reference table ─────────────────────────────────────────────
    if crossrefs:
        lines += [
            "",
            "### Cross-Reference with Management Claims",
            "",
            "| Alignment | Platform | Market | Prob. | Category | Claim | Interpretation |",
            "|-----------|----------|--------|-------|----------|-------|----------------|",
        ]
        for r in crossrefs:
            prob_str  = f"{r.yes_probability:.0%}"
            q_display = _escape_pipe(r.market_question)
            if r.url:
                q_display = f"[{q_display}]({r.url})"
            lines.append(
                f"| **{r.alignment}** | {r.platform} | {q_display} | {prob_str} "
                f"| {r.claim_category} | {_escape_pipe(r.claim_text)} "
                f"| {_escape_pipe(r.interpretation)} |"
            )

        contra = sum(1 for r in crossrefs if r.alignment == "CONTRADICTS")
        if contra:
            lines += [
                "",
                f"> **{contra} CONTRADICTS signal(s) detected.** "
                "Prediction markets are diverging from management guidance on the above "
                "topics. These represent areas of elevated information asymmetry and "
                "warrant independent verification before relying on management's narrative.",
            ]

    return "\n".join(lines)


def _render_backtest_context(bt: "PostEarningsReturns") -> str:
    """Render the Backtest Context section."""
    def _fmt(r: Optional[float]) -> str:
        if r is None:
            return "—"
        sign = "+" if r > 0 else ""
        return f"{sign}{r * 100:.1f}%"

    lines = [
        "## Backtest Context",
        "",
        "| Window | Return |",
        "|--------|--------|",
        f"| 1-day post-earnings  | {_fmt(bt.ret_1d)} |",
        f"| 5-day post-earnings  | {_fmt(bt.ret_5d)} |",
        f"| 20-day post-earnings | {_fmt(bt.ret_20d)} |",
        "",
        f"*Based on earnings call date {bt.call_date}. "
        "Retrospective data only — not a trading signal.*",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_report(
    company:            str,
    now_period:         str,
    prev_period:        str,
    changes:            list[Change],
    stats:              ReportStats | None = None,
    ai_sensitivity_md:  str = "",
    abandoned_metrics:  "list[AbandonedMetric] | None" = None,
    hedge_deltas:       "list[HedgeDelta] | None" = None,
    peer_signals:       "list[PeerSignal] | None" = None,
    backtest_returns:   "PostEarningsReturns | None" = None,
    pred_markets:       "list[PredictionMarket] | None" = None,
    pred_crossref:      "list[MarketClaimCrossRef] | None" = None,
) -> str:
    """Render a complete Markdown report string from a list of Changes.

    Args:
        company:           Company identifier (e.g. "BA").
        now_period:        Label for the current quarter (e.g. "2025Q4").
        prev_period:       Label for the prior quarter (e.g. "2025Q3").
        changes:           Output of diff.match_claims(), sorted by severity DESC.
        stats:             Optional coverage statistics; uses empty defaults if None.
        ai_sensitivity_md: Optional pre-rendered AI Sensitivity section Markdown.
        abandoned_metrics: Optional output of diff.find_abandoned_metrics().
        hedge_deltas:      Optional output of hedge_score.diff_hedge_scores().
        peer_signals:      Optional output of peer_contagion.load_peer_signals().
        backtest_returns:  Optional output of backtest.compute_post_earnings_returns().

    Returns:
        Complete Markdown document as a string.
    """
    if stats is None:
        stats = ReportStats()

    # Normalize None → empty list for section renderers
    _abandoned  = abandoned_metrics if abandoned_metrics is not None else []
    _hedge      = hedge_deltas      if hedge_deltas      is not None else []
    _peers      = peer_signals      if peer_signals      is not None else []
    _p_markets  = pred_markets      if pred_markets      is not None else []
    _p_crossref = pred_crossref     if pred_crossref     is not None else []

    # Section order:
    #   Header → Executive Summary → Red Flags → Abandoned Metrics
    #   → Hedging Intensity → Peer Signals
    #   → AI Sensitivity → Prediction Market Context
    #   → Backtest Context → Limitations → Methodology
    parts: list[str] = [
        _render_header(company, now_period, prev_period, changes, stats),
        _render_executive_summary(changes),
        _render_red_flags_table(changes),
        _render_abandoned_metrics(_abandoned),
        _render_hedging_intensity(_hedge),
        _render_peer_signals(_peers),
    ]

    if ai_sensitivity_md.strip():
        parts.append(ai_sensitivity_md.strip())

    pm_section = _render_prediction_markets(_p_markets, _p_crossref)
    if pm_section:
        parts.append(pm_section)

    if backtest_returns is not None:
        parts.append(_render_backtest_context(backtest_returns))

    parts.extend([
        _render_limitations(),
        _render_methodology(),
    ])

    # Filter out empty sections before joining
    return "\n\n---\n\n".join(p for p in parts if p) + "\n"


def save_report(
    report_md:   str,
    company:     str,
    now_period:  str,
    prev_period: str,
    output_dir:  Path,
) -> Path:
    """Write the Markdown report to *output_dir* and return the file path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{company}_{now_period}_vs_{prev_period}.md"
    out_path = output_dir / filename
    out_path.write_text(report_md, encoding="utf-8")
    logger.info("Report saved → %s", out_path)
    return out_path
