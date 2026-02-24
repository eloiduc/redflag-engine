from __future__ import annotations

import json
import logging
import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from segment import Chunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Category(str, Enum):
    guidance              = "guidance"
    demand                = "demand"
    pricing_margin        = "pricing_margin"
    liquidity             = "liquidity"
    reg_legal             = "reg_legal"
    competition           = "competition"
    costs_restructuring   = "costs_restructuring"
    ops_execution         = "ops_execution"
    accounting            = "accounting"
    other                 = "other"


class Polarity(str, Enum):
    positive = "positive"
    negative = "negative"
    neutral  = "neutral"
    mixed    = "mixed"


class Confidence(str, Enum):
    low    = "low"
    medium = "medium"
    high   = "high"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category:     Category
    polarity:     Polarity
    claim:        str
    evidence:     str
    chunk_id:     str
    confidence:   Confidence
    speaker_role: str = "unknown"   # set post-parse from Chunk; not in LLM schema

    @field_validator("evidence")
    @classmethod
    def evidence_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("evidence must not be empty")
        return v


class ClaimList(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claims: list[Claim]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a financial analyst assistant. Your ONLY job is to extract factual claims directly stated in the provided transcript chunk.

RULES (non-negotiable):
1. Extract ONLY claims that are explicitly supported by text in the chunk. Do not invent, infer, or extrapolate.
2. Every claim MUST include an evidence field: a VERBATIM quote copied character-for-character from the chunk, 25 words or fewer.
3. The evidence string must appear EXACTLY as written in the chunk — do not paraphrase, truncate mid-word, or alter punctuation.
4. If no claims can be extracted with proper verbatim evidence, return {"claims": []}.
5. Return at most 6 claims per chunk. Prioritise the most material statements.
6. Do NOT predict stock prices, returns, or market outcomes.
7. Do NOT add commentary, warnings, or explanation outside the JSON.
8. Output ONLY valid JSON matching the schema provided.\
"""

_USER_PROMPT_TEMPLATE = """\
Transcript chunk ID: {chunk_id}
Inferred section: {section}
Speaker role: {speaker_role}

--- BEGIN CHUNK ---
{chunk_text}
--- END CHUNK ---

Extract claims from this chunk following the system rules.
Each evidence value must be copied verbatim from the chunk above (max 25 words).
Return JSON with this exact schema:
{{
  "claims": [
    {{
      "category": "<one of: guidance|demand|pricing_margin|liquidity|reg_legal|competition|costs_restructuring|ops_execution|accounting|other>",
      "polarity": "<positive|negative|neutral|mixed>",
      "claim": "<atomic factual statement grounded in this chunk>",
      "evidence": "<verbatim quote copied exactly from the chunk above, max 25 words>",
      "chunk_id": "{chunk_id}",
      "confidence": "<low|medium|high>"
    }}
  ]
}}\
"""

MAX_CLAIMS_PER_CHUNK = 6
MAX_EVIDENCE_WORDS   = 25
MIN_CLAIM_WORDS      = 7   # shorter claims are almost always boilerplate

# Patterns that identify corporate header / speaker-intro lines
_TITLE_PATTERN = re.compile(
    r"\b(CEO|CFO|COO|CTO|CMO|President|Officer|Director|Chairman|Treasurer)\b",
    re.IGNORECASE,
)
_CORP_SUFFIX_PATTERN = re.compile(
    r"\b(Inc\.|Holdings|Corporation|Corp\.|Ltd\.?|LLC|PLC|S\.A\.)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_user_prompt(chunk: Chunk) -> str:
    return _USER_PROMPT_TEMPLATE.format(
        chunk_id=chunk.chunk_id,
        section=chunk.section,
        speaker_role=chunk.speaker_role,
        chunk_text=chunk.text,
    )


def _word_count(text: str) -> int:
    return len(text.split())


def _filter_claims(
    claims: list[Claim],
    chunk_id: str,
    chunk_text: str,
) -> list[Claim]:
    """Apply post-LLM validation and return only fully-passing claims.

    Checks (in order, all must pass):
    1. Evidence is non-empty.
    2. Evidence word count ≤ 25.
    3. Evidence is a verbatim substring of the chunk text (exact match).

    Claims failing any check are dropped and the reason is logged.
    The returned list is capped at MAX_CLAIMS_PER_CHUNK.
    """
    valid: list[Claim] = []
    for claim in claims:
        ev = claim.evidence.strip()
        cl = claim.claim.strip()

        # Check 0a — minimum claim length (drops boilerplate like speaker intros)
        if len(cl.split()) < MIN_CLAIM_WORDS:
            logger.debug(
                "%s: DROP claim too short (%d words) | claim: %.60s",
                chunk_id, len(cl.split()), cl,
            )
            continue

        # Check 0b — corporate header pattern (title + company suffix in same claim)
        if _TITLE_PATTERN.search(cl) and _CORP_SUFFIX_PATTERN.search(cl):
            logger.debug(
                "%s: DROP boilerplate header claim | claim: %.80s", chunk_id, cl,
            )
            continue

        # Check 1 — non-empty (also caught by Pydantic validator, belt-and-suspenders)
        if not ev:
            logger.warning("%s: DROP empty evidence | claim: %.60s", chunk_id, claim.claim)
            continue

        # Check 2 — word count
        wc = _word_count(ev)
        if wc > MAX_EVIDENCE_WORDS:
            logger.warning(
                "%s: DROP evidence too long (%d words) | evidence: %.60s…",
                chunk_id, wc, ev,
            )
            continue

        # Check 3 — verbatim substring in chunk (strict exact match)
        if ev not in chunk_text:
            logger.warning(
                "%s: DROP evidence not verbatim in chunk | evidence: %.80s",
                chunk_id, ev,
            )
            continue

        # Enforce correct chunk_id regardless of what the model returned.
        if claim.chunk_id != chunk_id:
            claim = claim.model_copy(update={"chunk_id": chunk_id})

        valid.append(claim)
        if len(valid) == MAX_CLAIMS_PER_CHUNK:
            break

    return valid


def _parse_response(raw: str, chunk_id: str, chunk_text: str) -> list[Claim]:
    """Parse + validate the raw JSON string from the API response."""
    text = raw.strip()
    # Strip markdown code fences if the model wrapped its output.
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            line for line in lines if not line.startswith("```")
        ).strip()

    try:
        data: Any = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.error("%s: JSON parse error — %s", chunk_id, exc)
        return []

    try:
        claim_list = ClaimList.model_validate(data)
    except Exception as exc:
        logger.error("%s: Pydantic validation error — %s", chunk_id, exc)
        return []

    return _filter_claims(claim_list.claims, chunk_id, chunk_text)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _call_api(chunk: Chunk, client: Any) -> str | None:
    """Call the Claude API with one retry on failure. Returns raw text or None."""
    user_prompt = _build_user_prompt(chunk)
    kwargs = dict(
        model="claude-opus-4-6",
        max_tokens=1024,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    for attempt in (1, 2):
        try:
            response = client.messages.create(**kwargs)
            return response.content[0].text
        except Exception as exc:
            if attempt == 1:
                logger.warning(
                    "%s: API error on attempt 1 (%s), retrying…", chunk.chunk_id, exc
                )
            else:
                logger.warning(
                    "%s: API error on attempt 2 (%s), skipping chunk", chunk.chunk_id, exc
                )
    return None


def extract_claims_from_chunk(chunk: Chunk, client: Any) -> list[Claim]:
    """Call the Claude API for a single chunk and return validated Claims.

    Args:
        chunk:  The transcript chunk to analyse.
        client: An instantiated ``anthropic.Anthropic`` client.

    Returns:
        List of validated Claim objects (may be empty).
    """
    raw = _call_api(chunk, client)
    if raw is None:
        return []

    claims = _parse_response(raw, chunk.chunk_id, chunk.text)

    # Propagate speaker_role from chunk onto each validated claim.
    if chunk.speaker_role != "unknown":
        claims = [c.model_copy(update={"speaker_role": chunk.speaker_role})
                  for c in claims]

    logger.debug(
        "%s: extracted %d claim(s)  section=%s  speaker=%s",
        chunk.chunk_id, len(claims), chunk.section, chunk.speaker_role,
    )
    return claims


def extract_claims(chunks: list[Chunk], client: Any) -> list[Claim]:
    """Extract claims from all chunks, skipping any that fail.

    Args:
        chunks: Ordered list of Chunk objects from segment_doc().
        client: An instantiated ``anthropic.Anthropic`` client.

    Returns:
        Flat list of all validated Claims across the document.
    """
    all_claims: list[Claim] = []
    for chunk in chunks:
        claims = extract_claims_from_chunk(chunk, client)
        all_claims.extend(claims)
    logger.info(
        "Total claims extracted: %d from %d chunks", len(all_claims), len(chunks)
    )
    return all_claims
