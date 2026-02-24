"""peer_contagion.py — Peer and supplier signal propagation.

Reads existing Red Flag reports from outputs/ for companies related to the
target ticker (as defined in peer_map.json) and surfaces red flags that may
indicate sector-level or supply-chain stress.

No LLM calls — fully deterministic, reads existing .md report files.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

# Maximum number of peer signals to surface (suppliers take priority slots)
_MAX_SIGNALS = 10

# Severity levels considered "notable" for NEW claims
_HIGH_SEV_LABELS: frozenset[str] = frozenset({"Critical", "High"})


# ---------------------------------------------------------------------------
# Pydantic model
# ---------------------------------------------------------------------------

class PeerSignal(BaseModel):
    """A single red flag sourced from a related company's report."""

    model_config = ConfigDict(extra="forbid")

    source_company:  str   # ticker of the peer / supplier
    relationship:    str   # "peer" | "supplier"
    category:        str
    claim:           str
    evidence:        str
    polarity:        str   # "negative" | "mixed"
    sev:             str   # severity label, e.g. "High", "Critical"
    report_filename: str


# ---------------------------------------------------------------------------
# Minimal Markdown table parser
# (cannot import _parse_md_table from streamlit_app.py — lives outside src/)
# ---------------------------------------------------------------------------

def _split_pipe_row(line: str) -> list[str]:
    """Split one Markdown table row on unescaped pipes."""
    parts = re.split(r"(?<!\\)\|", line)
    # Drop leading/trailing empty cells from the outer | delimiters
    while parts and not parts[0].strip():
        parts.pop(0)
    while parts and not parts[-1].strip():
        parts.pop()
    return [p.strip().replace("\\|", "|") for p in parts]


def _parse_red_flags_table(md_text: str) -> list[dict[str, str]]:
    """Extract rows from the ## Red Flags section of a report Markdown string.

    Returns a list of dicts keyed by the table header names.
    Returns an empty list if the section or table is absent / malformed.
    """
    # Split report into sections using the --- separator
    parts = re.split(r"\n\n---\n\n", md_text)
    for part in parts:
        if re.match(r"^## Red Flags\b", part.strip()):
            # Found the Red Flags section — parse the pipe table within it
            lines = [
                ln for ln in part.splitlines()
                if re.match(r"^\s*\|", ln)
            ]
            if len(lines) < 3:
                return []
            headers = _split_pipe_row(lines[0])
            rows: list[dict[str, str]] = []
            for line in lines[2:]:  # skip header + separator row
                cells = _split_pipe_row(line)
                if len(cells) == len(headers):
                    rows.append(dict(zip(headers, cells)))
            return rows
    return []


# ---------------------------------------------------------------------------
# Polarity proxy
# ---------------------------------------------------------------------------

def _polarity_proxy(change: str, sev: str) -> str | None:
    """Map a Change cell + Sev cell to a polarity string, or None to skip.

    Rules:
      - "WORSENED" in change → "negative"
      - "NEW" in change AND sev in _HIGH_SEV_LABELS → "mixed"
      - All other rows → None (excluded)
    """
    if "WORSENED" in change:
        return "negative"
    if "NEW" in change and sev in _HIGH_SEV_LABELS:
        return "mixed"
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_peer_signals(
    company:       str,
    outputs_dir:   Path,
    peer_map_path: Path,
) -> list[PeerSignal]:
    """Load red flag signals from related companies' existing reports.

    Args:
        company:       Target company ticker (e.g. "BA").
        outputs_dir:   Directory containing generated .md reports.
        peer_map_path: Path to peer_map.json.

    Returns:
        List of :class:`PeerSignal` sorted by (supplier-first, sev DESC).
        Returns an empty list gracefully on any configuration or parse error.
    """
    # ── Load peer map ─────────────────────────────────────────────────────
    if not peer_map_path.exists():
        logger.warning("peer_map.json not found at %s — skipping peer signals", peer_map_path)
        return []

    try:
        with peer_map_path.open(encoding="utf-8") as fh:
            peer_map: dict[str, dict[str, list[str]]] = json.load(fh)
    except Exception as exc:
        logger.warning("Failed to load peer_map.json: %s", exc)
        return []

    company_upper = company.upper()
    entry = peer_map.get(company_upper)
    if entry is None:
        logger.info("No peer map entry for %s — skipping peer signals", company_upper)
        return []

    suppliers: list[str] = entry.get("suppliers", [])
    peers:     list[str] = entry.get("peers",     [])

    # Process suppliers first so they get priority slots
    ordered: list[tuple[str, str]] = (
        [(t, "supplier") for t in suppliers]
        + [(t, "peer")     for t in peers]
    )

    # ── Collect signals ───────────────────────────────────────────────────
    signals: list[PeerSignal] = []

    for ticker, relationship in ordered:
        if len(signals) >= _MAX_SIGNALS:
            break

        # Find the most recent report for this ticker (lexicographic last)
        pattern = f"{ticker.upper()}_*.md"
        matches = sorted(outputs_dir.glob(pattern))
        if not matches:
            logger.debug("No reports found for %s (%s of %s)", ticker, relationship, company_upper)
            continue

        report_path = matches[-1]   # latest by filename sort
        try:
            md_text = report_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:
            logger.warning("Could not read %s: %s", report_path.name, exc)
            continue

        rows = _parse_red_flags_table(md_text)
        if not rows:
            logger.debug("No Red Flags table in %s", report_path.name)
            continue

        for row in rows:
            if len(signals) >= _MAX_SIGNALS:
                break

            change = row.get("Change", "")
            sev    = row.get("Sev",    "")
            polarity = _polarity_proxy(change, sev)
            if polarity is None:
                continue

            signals.append(PeerSignal(
                source_company  = ticker.upper(),
                relationship    = relationship,
                category        = row.get("Category", ""),
                claim           = row.get("Evidence (Now)", ""),
                evidence        = row.get("Evidence (Now)", ""),
                polarity        = polarity,
                sev             = sev,
                report_filename = report_path.name,
            ))

        logger.debug(
            "Processed %s (%s): %d rows → signals so far: %d",
            ticker, relationship, len(rows), len(signals),
        )

    logger.info(
        "Peer contagion: %d signals found for %s (%d peers/suppliers checked)",
        len(signals), company_upper, len(ordered),
    )
    return signals
