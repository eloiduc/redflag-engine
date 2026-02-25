"""hedge_score.py — Hedging language intensity scoring.

For each quarter's chunk list, counts Tier 1 and Tier 2 hedge terms per section
(normalized per 100 words), strips safe-harbour boilerplate paragraphs first.
Compares two quarters and flags sections where hedging increased by > 3 pp.

No LLM calls — fully deterministic regex/string matching.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from segment import Chunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hedge word tiers (static wordlists — no external data)
# ---------------------------------------------------------------------------

# Tier 1: strong hedges — single words and short phrases
_TIER1_WORDS: frozenset[str] = frozenset({
    "may", "might", "could", "uncertain", "unclear", "contingent",
})
_TIER1_PHRASES: tuple[str, ...] = (
    "subject to", "no assurance", "no guarantee", "cannot guarantee",
    "subject to change", "forward-looking", "risk factors",
)

# Tier 2: moderate hedges
_TIER2_WORDS: frozenset[str] = frozenset({
    "expect", "anticipate", "believe", "intend", "plan", "seek", "estimate",
    "approximately", "targeted", "potential", "possible",
    "likely", "assume", "assumed", "projected",
})
_TIER2_PHRASES: tuple[str, ...] = ()   # none for Tier 2 currently

# Tier 3: safe-harbour markers — paragraphs containing these are stripped
_TIER3_PHRASES: tuple[str, ...] = (
    "safe harbor",
    "safe harbour",
    "forward-looking statements",
    "actual results may differ",
    "speak only as of",
)

# Pre-compiled single-word patterns (word-boundary wrapped)
_TIER1_WORD_RES: tuple[re.Pattern[str], ...] = tuple(
    re.compile(r"\b" + re.escape(w) + r"\b", re.IGNORECASE) for w in _TIER1_WORDS
)
_TIER2_WORD_RES: tuple[re.Pattern[str], ...] = tuple(
    re.compile(r"\b" + re.escape(w) + r"\b", re.IGNORECASE) for w in _TIER2_WORDS
)

# Flag delta threshold
_FLAG_DELTA_PP: float = 3.0


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class HedgeDelta(BaseModel):
    """Comparison of hedge scores for one section between two quarters."""

    model_config = ConfigDict(extra="forbid")

    section:    str
    now_score:  float   # hedge matches per 100 words (current quarter)
    prev_score: float   # hedge matches per 100 words (prior quarter)
    delta:      float   # now_score − prev_score (positive = more hedged now)
    flag:       bool    # True when delta > _FLAG_DELTA_PP


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _strip_safe_harbour(text: str) -> str:
    """Remove paragraphs that contain a Tier 3 safe-harbour phrase."""
    paragraphs = re.split(r"\n{2,}", text)
    clean: list[str] = []
    for para in paragraphs:
        para_lower = para.lower()
        if any(phrase in para_lower for phrase in _TIER3_PHRASES):
            continue
        clean.append(para)
    return "\n\n".join(clean)


def _count_matches(text: str) -> int:
    """Count total Tier 1 + Tier 2 hedge term matches in *text*."""
    count = 0
    # Single-word patterns (word-boundary regex)
    for pat in _TIER1_WORD_RES:
        count += len(pat.findall(text))
    for pat in _TIER2_WORD_RES:
        count += len(pat.findall(text))
    # Multi-word phrases (simple case-insensitive substring)
    text_lower = text.lower()
    for phrase in _TIER1_PHRASES:
        count += text_lower.count(phrase)
    for phrase in _TIER2_PHRASES:
        count += text_lower.count(phrase)
    return count


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_hedging(chunks: "list[Chunk]") -> dict[str, float]:
    """Return a mapping of section label → hedge score (matches per 100 words).

    Safe-harbour paragraphs are stripped before counting.  Scores for chunks
    with the same section label are averaged.

    Args:
        chunks: Segmented transcript chunks (current or prior quarter).

    Returns:
        Dict keyed by section label (e.g. ``"guidance"``, ``"demand"``).
        Sections with zero total words after stripping are skipped.
    """
    # Accumulate (total_matches, total_words) per section
    section_data: dict[str, list[float]] = {}

    for chunk in chunks:
        text = _strip_safe_harbour(chunk.text)
        words = text.split()
        if not words:
            continue
        count = _count_matches(text)
        score = count / len(words) * 100.0
        section_data.setdefault(chunk.section, []).append(score)

    result: dict[str, float] = {}
    for section, scores in section_data.items():
        result[section] = round(sum(scores) / len(scores), 2)

    logger.debug("Hedge scores by section: %s", result)
    return result


def diff_hedge_scores(
    now_scores:  dict[str, float],
    prev_scores: dict[str, float],
) -> list[HedgeDelta]:
    """Compare section-level hedge scores between two quarters.

    For sections present only in one quarter, the missing quarter's score
    is treated as 0.0 (new section or dropped section).

    Args:
        now_scores:  Output of ``score_hedging()`` for the current quarter.
        prev_scores: Output of ``score_hedging()`` for the prior quarter.

    Returns:
        List of :class:`HedgeDelta` sorted by ``|delta|`` descending.
    """
    all_sections = set(now_scores) | set(prev_scores)
    deltas: list[HedgeDelta] = []

    for section in all_sections:
        now_val  = now_scores.get(section, 0.0)
        prev_val = prev_scores.get(section, 0.0)
        delta    = round(now_val - prev_val, 2)
        deltas.append(HedgeDelta(
            section    = section,
            now_score  = round(now_val,  2),
            prev_score = round(prev_val, 2),
            delta      = delta,
            flag       = abs(delta) > _FLAG_DELTA_PP,
        ))

    # Sort by absolute delta descending; flagged rows first on ties
    deltas.sort(key=lambda d: (-abs(d.delta), not d.flag))

    flagged = sum(1 for d in deltas if d.flag)
    logger.info(
        "Hedge diff: %d sections  flagged=%d  (threshold=±%.1f pp)",
        len(deltas), flagged, _FLAG_DELTA_PP,
    )
    return deltas
