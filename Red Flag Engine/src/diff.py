from __future__ import annotations

import logging
from collections import defaultdict
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict
from rapidfuzz.fuzz import token_set_ratio

from llm_extract import Category, Confidence, Claim, Polarity

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MATCH_THRESHOLD:      int = 65   # strict pass threshold
SOFT_THRESHOLD:       int = 60   # soft pass threshold (same-category, first 60 chars)
SOFT_WINDOW:          int = 60   # chars used for soft comparison

# Lower numeric value = more negative sentiment
POLARITY_ORDER: dict[Polarity, int] = {
    Polarity.negative: 0,
    Polarity.mixed:    1,
    Polarity.neutral:  2,
    Polarity.positive: 3,
}

CONF_ORDER: dict[Confidence, int] = {
    Confidence.low:    0,
    Confidence.medium: 1,
    Confidence.high:   2,
}

# Severity +1 for high-risk categories on NEW / WORSENED claims
_HIGH_RISK_CATEGORIES: frozenset[Category] = frozenset({
    Category.liquidity,
    Category.reg_legal,
    Category.accounting,
})

# Extra +1 for high-confidence worsened claims in financially sensitive categories
_HIGH_CONF_BUMP_CATEGORIES: frozenset[Category] = frozenset({
    Category.liquidity,
    Category.guidance,
    Category.reg_legal,
})


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ChangeType(str, Enum):
    new       = "new"
    worsened  = "worsened"
    improved  = "improved"
    unchanged = "unchanged"


class Change(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category:         Category
    change_type:      ChangeType
    severity:         int               # 1–5
    confidence:       Confidence
    summary:          str
    # Current quarter
    now_claim:        str
    now_evidence:     str
    now_chunk_id:     str
    now_speaker_role: str = "unknown"   # propagated from Claim.speaker_role
    # Prior quarter (None for new claims)
    prev_claim:       Optional[str] = None
    prev_evidence:    Optional[str] = None
    prev_chunk_id:    Optional[str] = None
    # Similarity metadata
    match_score:      float = 0.0
    match_quality:    str   = "strict"  # "strict" | "soft"


# ---------------------------------------------------------------------------
# Severity logic
# ---------------------------------------------------------------------------

def _assign_severity(
    change_type:  ChangeType,
    category:     Category,
    polarity_now: Polarity,
    confidence:   Confidence,
) -> int:
    """Return a severity score in [1, 5] applying all rules in order.

    Base rules:
      new       → 3  (+1 if high-risk category, +1 if negative polarity)
      worsened  → 4  (+1 if high-risk category)
      improved  → 1  (+1 if high confidence)
      unchanged → 2

    Post-base adjustments:
      low confidence  → cap at 3
      high confidence + worsened + bump-category → +1 (cap 5)
    """
    # ── Base ──────────────────────────────────────────────────────────────
    if change_type == ChangeType.new:
        base = 3
        if category in _HIGH_RISK_CATEGORIES:
            base += 1
        if polarity_now == Polarity.negative:
            base += 1

    elif change_type == ChangeType.worsened:
        base = 4
        if category in _HIGH_RISK_CATEGORIES:
            base += 1

    elif change_type == ChangeType.improved:
        base = 1
        if confidence == Confidence.high:
            base += 1

    else:  # unchanged
        base = 2

    base = min(base, 5)

    # ── Confidence adjustments ────────────────────────────────────────────
    if confidence == Confidence.low:
        base = min(base, 3)

    if (confidence == Confidence.high
            and change_type == ChangeType.worsened
            and category in _HIGH_CONF_BUMP_CATEGORIES):
        base = min(base + 1, 5)

    return base


# ---------------------------------------------------------------------------
# Change detection helpers
# ---------------------------------------------------------------------------

def _determine_change_type(
    polarity_now:  Polarity,
    polarity_prev: Polarity,
) -> ChangeType:
    """Map a polarity transition to a ChangeType."""
    now_val  = POLARITY_ORDER[polarity_now]
    prev_val = POLARITY_ORDER[polarity_prev]
    if now_val < prev_val:
        return ChangeType.worsened
    if now_val > prev_val:
        return ChangeType.improved
    return ChangeType.unchanged


def _build_summary(
    change_type: ChangeType,
    claim_now:   Claim,
    claim_prev:  Optional[Claim],
) -> str:
    """Produce a one-line human-readable summary for a Change."""
    cat = claim_now.category.value.replace("_", " ").title()

    if change_type == ChangeType.new:
        return f"[NEW] {cat}: {claim_now.claim}"
    if change_type == ChangeType.worsened:
        return (
            f"[WORSENED] {cat}: sentiment shifted "
            f"{claim_prev.polarity.value} → {claim_now.polarity.value}. "
            f"{claim_now.claim}"
        )
    if change_type == ChangeType.improved:
        return (
            f"[IMPROVED] {cat}: sentiment shifted "
            f"{claim_prev.polarity.value} → {claim_now.polarity.value}. "
            f"{claim_now.claim}"
        )
    return f"[UNCHANGED] {cat}: {claim_now.claim}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def match_claims(
    now_claims:  list[Claim],
    prev_claims: list[Claim],
    threshold:   int = MATCH_THRESHOLD,
) -> list[Change]:
    """Compare two claim lists and return classified, ranked Changes.

    Two-pass matching strategy
    --------------------------
    Pass 1 (strict):  token_set_ratio on full claim text >= threshold.
    Pass 2 (soft):    For unmatched claims only — token_set_ratio on the
                      first SOFT_WINDOW chars, same category, >= SOFT_THRESHOLD.
                      Produces match_quality="soft".

    Args:
        now_claims:  Claims from the current (newer) quarter.
        prev_claims: Claims from the prior quarter. May be empty.
        threshold:   Strict match threshold (default 72).

    Returns:
        List of Change objects sorted by severity DESC, then confidence DESC.
    """
    changes: list[Change] = []

    for now in now_claims:
        # ── Pass 1: strict full-text match ─────────────────────────────
        best_score: float = 0.0
        best_prev:  Optional[Claim] = None

        for prev in prev_claims:
            score = token_set_ratio(now.claim, prev.claim)
            if score > best_score:
                best_score = score
                best_prev  = prev

        if best_score >= threshold and best_prev is not None:
            match_quality = "strict"
        else:
            # ── Pass 2: soft same-category short-window match ───────────
            soft_score: float = 0.0
            soft_prev:  Optional[Claim] = None

            for prev in prev_claims:
                if prev.category != now.category:
                    continue
                score = token_set_ratio(
                    now.claim[:SOFT_WINDOW], prev.claim[:SOFT_WINDOW]
                )
                if score > soft_score:
                    soft_score = score
                    soft_prev  = prev

            if soft_score >= SOFT_THRESHOLD and soft_prev is not None:
                best_score    = soft_score
                best_prev     = soft_prev
                match_quality = "soft"
                logger.debug(
                    "%s: soft match score=%.1f cat=%s",
                    now.chunk_id, soft_score, now.category.value,
                )
            else:
                match_quality = "strict"   # new claim — no match either pass
                best_prev     = None

        # ── Classify ───────────────────────────────────────────────────
        if best_prev is not None:
            change_type   = _determine_change_type(now.polarity, best_prev.polarity)
            prev_claim    = best_prev.claim
            prev_evidence = best_prev.evidence
            prev_chunk_id = best_prev.chunk_id
        else:
            change_type   = ChangeType.new
            prev_claim    = None
            prev_evidence = None
            prev_chunk_id = None

        severity = _assign_severity(change_type, now.category, now.polarity, now.confidence)
        summary  = _build_summary(change_type, now, best_prev)

        changes.append(Change(
            category=now.category,
            change_type=change_type,
            severity=severity,
            confidence=now.confidence,
            summary=summary,
            now_claim=now.claim,
            now_evidence=now.evidence,
            now_chunk_id=now.chunk_id,
            now_speaker_role=now.speaker_role,
            prev_claim=prev_claim,
            prev_evidence=prev_evidence,
            prev_chunk_id=prev_chunk_id,
            match_score=round(best_score, 1),
            match_quality=match_quality,
        ))

        logger.debug(
            "%s → %s  sev=%d  score=%.1f  quality=%s  cat=%s",
            now.chunk_id, change_type.value, severity,
            best_score, match_quality, now.category.value,
        )

    # Sort: severity descending, then confidence descending
    changes.sort(key=lambda c: (-c.severity, -CONF_ORDER[c.confidence]))

    n_new       = sum(1 for c in changes if c.change_type == ChangeType.new)
    n_worsened  = sum(1 for c in changes if c.change_type == ChangeType.worsened)
    n_improved  = sum(1 for c in changes if c.change_type == ChangeType.improved)
    n_unchanged = sum(1 for c in changes if c.change_type == ChangeType.unchanged)
    n_soft      = sum(1 for c in changes if c.match_quality == "soft")

    logger.info(
        "Diff: %d changes  new=%d  worsened=%d  improved=%d  unchanged=%d  soft=%d",
        len(changes), n_new, n_worsened, n_improved, n_unchanged, n_soft,
    )

    return changes


# ---------------------------------------------------------------------------
# Abandoned metric detection
# ---------------------------------------------------------------------------

class AbandonedMetric(BaseModel):
    """A claim category that was prominent last quarter but absent this quarter."""

    model_config = ConfigDict(extra="forbid")

    category:             Category
    representative_claim: str    # most-confident prev claim text for this category
    evidence:             str    # evidence quote from the representative claim
    chunk_id:             str    # chunk_id of the representative claim
    confidence:           Confidence  # high ≥3 prev claims, medium ≥2


def find_abandoned_metrics(
    claims_now:  list[Claim],
    claims_prev: list[Claim],
    threshold:   int = MATCH_THRESHOLD,
) -> list[AbandonedMetric]:
    """Identify claim categories present last quarter but absent this quarter.

    A category is "abandoned" when:
      - It has ≥ 2 prior-quarter claims, AND
      - It is not an all-positive, non-guidance category (these may drop off
        naturally without being a signal), AND
      - No current-quarter claim has token_set_ratio ≥ SOFT_THRESHOLD against
        any prior-quarter claim in that category.

    Args:
        claims_now:  Claims extracted from the current quarter.
        claims_prev: Claims extracted from the prior quarter.
        threshold:   Ignored (kept for API symmetry with match_claims).

    Returns:
        List of :class:`AbandonedMetric` sorted by confidence DESC then
        category name ASC.  Empty list if claims_prev is empty.
    """
    if not claims_prev:
        return []

    # ── Group prior claims by category ───────────────────────────────────
    by_cat: dict[Category, list[Claim]] = defaultdict(list)
    for c in claims_prev:
        by_cat[c.category].append(c)

    abandoned: list[AbandonedMetric] = []

    for cat, prev_claims_in_cat in by_cat.items():
        # ── Skip if too few prior claims ──────────────────────────────────
        if len(prev_claims_in_cat) < 2:
            continue

        # ── Skip all-positive, non-guidance categories ─────────────────────
        # Positive signals naturally drop off — not a red flag unless guidance
        all_positive = all(c.polarity == Polarity.positive for c in prev_claims_in_cat)
        if all_positive and cat != Category.guidance:
            continue

        # ── Check for any current-quarter match above SOFT_THRESHOLD ─────
        category_abandoned = True
        for prev_c in prev_claims_in_cat:
            for now_c in claims_now:
                score = token_set_ratio(prev_c.claim, now_c.claim)
                if score >= SOFT_THRESHOLD:
                    category_abandoned = False
                    break
            if not category_abandoned:
                break

        if not category_abandoned:
            continue

        # ── Pick representative claim: highest confidence, then longest ───
        rep = max(
            prev_claims_in_cat,
            key=lambda c: (CONF_ORDER[c.confidence], len(c.claim)),
        )

        # ── Assign confidence based on count ─────────────────────────────
        count = len(prev_claims_in_cat)
        if count >= 3:
            conf = Confidence.high
        else:
            conf = Confidence.medium   # count == 2 by the ≥2 guard above

        abandoned.append(AbandonedMetric(
            category             = cat,
            representative_claim = rep.claim,
            evidence             = rep.evidence,
            chunk_id             = rep.chunk_id,
            confidence           = conf,
        ))

    # Sort: confidence DESC, then category name ASC
    abandoned.sort(key=lambda m: (-CONF_ORDER[m.confidence], m.category.value))

    logger.info(
        "Abandoned metrics: %d categories flagged (prev had %d categories with ≥2 claims)",
        len(abandoned),
        sum(1 for cl in by_cat.values() if len(cl) >= 2),
    )
    return abandoned
