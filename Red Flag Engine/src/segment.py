from __future__ import annotations

import re
import logging
from dataclasses import dataclass

from ingest import Doc

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Section inference — first matching pattern wins; fallback is "general"
# ---------------------------------------------------------------------------

SECTION_HINTS: list[tuple[str, str]] = [
    ("guidance",              r"\b(guid(ance)?|outlook|forecast|full.?year|target)\b"),
    ("liquidity",             r"\b(cash|liquidity|debt|leverage|balance.?sheet|credit.?facility)\b"),
    ("demand",                r"\b(demand|volume|unit|backlog|order|customer)\b"),
    ("pricing_margin",        r"\b(pric(e|ing)|margin|gross|spread|ASP)\b"),
    ("reg_legal",             r"\b(regulat|compliance|legal|lawsuit|SEC|DOJ|EPA)\b"),
    ("competition",           r"\b(compet|market.?share|rival|peer)\b"),
    ("costs_restructuring",   r"\b(cost|restructur|headcount|layoff|reorg|savings)\b"),
]

# Pre-compile for speed.
_COMPILED_HINTS: list[tuple[str, re.Pattern[str]]] = [
    (label, re.compile(pattern, re.IGNORECASE))
    for label, pattern in SECTION_HINTS
]

_SEPARATOR = "\n\n"
_SEP_LEN   = len(_SEPARATOR)   # 2

# ---------------------------------------------------------------------------
# Speaker-role inference — regex-based, no LLM call
# ---------------------------------------------------------------------------

_SPEAKER_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # operator: line starts with "OPERATOR" or "Operator"
    ("operator", re.compile(r"(?m)^OPERATOR\b|^Operator\b")),
    # analyst: "your next question", "next question comes from", or
    # a capitalised name followed by a colon at line start (analyst intro lines)
    ("analyst",  re.compile(
        r"(?mi)^[A-Z][A-Z\s\-']+:\s|"          # e.g. "JOHN DOE: ..."
        r"your next question|"
        r"next question comes from|"
        r"our next question|"
        r"\banalyst\b"
    )),
    # management: CFO/CEO/President/VP titles, or "Thank you, operator" opener
    ("management", re.compile(
        r"(?mi)\b(CFO|CEO|President|COO|CTO|EVP|SVP|VP|"
        r"Chief Financial|Chief Executive|Chief Operating)\b|"
        r"(?m)^Thank you,?\s+[Oo]perator"
    )),
]


def tag_speaker_role(text: str) -> str:
    """Infer the dominant speaker role in a chunk using regex patterns.

    Returns one of: ``"operator"``, ``"analyst"``, ``"management"``,
    ``"unknown"``.  First match wins in the priority order above.
    """
    for role, pattern in _SPEAKER_PATTERNS:
        if pattern.search(text):
            return role
    return "unknown"


@dataclass
class Chunk:
    chunk_id:     str   # "chunk_000", "chunk_001", …
    section:      str   # inferred section label
    text:         str
    speaker_role: str = "unknown"   # operator | analyst | management | unknown


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def infer_section(text: str) -> str:
    """Return the first matching section label for *text*, or 'general'."""
    for label, pattern in _COMPILED_HINTS:
        if pattern.search(text):
            return label
    return "general"


def chunk_text(text: str, max_chars: int = 3500) -> list[str]:
    """Split *text* into chunks of at most *max_chars* characters.

    Splits preferentially on blank lines (paragraph boundaries).  The running
    length correctly accounts for the ``\\n\\n`` separator that will be inserted
    between paragraphs when the chunk is joined.  A single paragraph that
    exceeds *max_chars* is hard-split at the character limit rather than
    being dropped.
    """
    paragraphs = re.split(r"\n{2,}", text)

    chunks: list[str] = []
    current_parts: list[str] = []
    current_len: int = 0   # tracks *actual* joined byte length including separators

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # If a single paragraph is itself too long, hard-split it first.
        if len(para) > max_chars:
            if current_parts:
                chunks.append(_SEPARATOR.join(current_parts))
                current_parts = []
                current_len = 0
            for i in range(0, len(para), max_chars):
                chunks.append(para[i : i + max_chars])
            continue

        # Cost of adding this paragraph: the paragraph itself plus the
        # separator that would precede it (if there are already parts).
        sep_cost = _SEP_LEN if current_parts else 0
        if current_len + sep_cost + len(para) > max_chars and current_parts:
            chunks.append(_SEPARATOR.join(current_parts))
            current_parts = []
            current_len = 0
            sep_cost = 0

        current_parts.append(para)
        current_len += (0 if len(current_parts) == 1 else _SEP_LEN) + len(para)

    if current_parts:
        chunks.append(_SEPARATOR.join(current_parts))

    return chunks


def segment_doc(doc: Doc, max_chars: int = 3500) -> list[Chunk]:
    """Segment a Doc into labeled Chunk objects.

    Args:
        doc:       The loaded transcript document.
        max_chars: Maximum characters per chunk (default 3500).

    Returns:
        Ordered list of Chunk objects with unique chunk_ids.
    """
    raw_chunks = chunk_text(doc.text, max_chars=max_chars)

    chunks: list[Chunk] = []
    for idx, text in enumerate(raw_chunks):
        chunk_id     = f"chunk_{idx:03d}"
        section      = infer_section(text)
        speaker_role = tag_speaker_role(text)
        chunks.append(Chunk(chunk_id=chunk_id, section=section, text=text,
                            speaker_role=speaker_role))
        logger.debug("  %s  section=%-22s  len=%d", chunk_id, section, len(text))

    logger.info(
        "Segmented '%s %s' → %d chunks", doc.company, doc.period, len(chunks)
    )
    return chunks
