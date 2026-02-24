from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from diff import Change, ChangeType

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

_MONITOR_CHECKLIST = [
    "**Guidance** — Did management revise full-year targets up or down?",
    "**Demand** — Are volume, backlog, or order trends deteriorating?",
    "**Margins** — Are gross or operating margins compressing?",
    "**Liquidity** — Has the cash position, leverage, or credit facility changed materially?",
    "**Regulatory / Legal** — Are there new compliance issues, lawsuits, or regulatory inquiries?",
]

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


def _render_monitor_checklist() -> str:
    lines = ["## Monitor Checklist", ""]
    for item in _MONITOR_CHECKLIST:
        lines.append(f"- [ ] {item}")
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
        "low-confidence claims are capped at severity 3.",
    ])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_report(
    company:     str,
    now_period:  str,
    prev_period: str,
    changes:     list[Change],
    stats:       ReportStats | None = None,
) -> str:
    """Render a complete Markdown report string from a list of Changes.

    Args:
        company:     Company identifier (e.g. "BA").
        now_period:  Label for the current quarter (e.g. "2025Q4").
        prev_period: Label for the prior quarter (e.g. "2025Q3").
        changes:     Output of diff.match_claims(), sorted by severity DESC.
        stats:       Optional coverage statistics; uses empty defaults if None.

    Returns:
        Complete Markdown document as a string.
    """
    if stats is None:
        stats = ReportStats()

    sections = [
        _render_header(company, now_period, prev_period, changes, stats),
        _render_executive_summary(changes),
        _render_red_flags_table(changes),
        _render_monitor_checklist(),
        _render_limitations(),
        _render_methodology(),
    ]
    return "\n\n---\n\n".join(sections) + "\n"


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
