"""ai_sensitivity.py — AI announcement sensitivity analysis.

For each analysed company, this module makes a single LLM call to assess how
sensitive the stock is to AI-related announcements (OpenAI, Anthropic, Google,
Microsoft, Meta, xAI, etc.) — both on the downside and the upside.

The analysis is grounded in:
  - The company's industry and competitive position
  - Claims extracted from the current-quarter earnings transcript
  - Structural dynamics of AI-driven disruption and adoption
"""

from __future__ import annotations

import logging
from typing import Any

from llm_extract import Claim

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a senior equity analyst at a quantitative hedge fund. Your area of \
specialisation is technology sector disruption and the market-moving impact of \
AI announcements on individual equities.

Your task: assess how sensitive a given company's stock is to announcements \
from major frontier AI organisations — OpenAI, Anthropic, Google DeepMind, \
Meta AI, Microsoft, xAI, Mistral, and similar — covering model releases, \
capability breakthroughs, pricing changes, API availability, partnerships, \
and regulatory developments.

You will receive:
  1. The company ticker
  2. A structured list of claims extracted from the company's most recent \
earnings call (category, polarity, and statement)

Your analysis MUST:
- Identify the company's industry and the precise channels through which AI \
  announcements affect it
- Distinguish between companies that are NET BENEFICIARIES of AI progress \
  (infrastructure, picks-and-shovels, enterprise adoption plays) and those \
  that are NET THREATENED (products being commoditised, workflows automated, \
  TAM shrinking)
- Provide specific, mechanistic reasoning for each risk or opportunity — not \
  vague generalities
- Reference precedents where AI announcements caused material stock moves \
  (e.g. Chegg -40% post-ChatGPT, NVDA +15% post-GPT-4, Veeva +8% after \
  AI EHR speculation)
- Assign a 5-level sensitivity rating with clear justification
- Remain strictly factual — do NOT predict returns or stock prices

Output ONLY the Markdown section below. No preamble. No text outside \
the Markdown structure. Replace all placeholder text.

---

## AI Announcement Sensitivity

**Sensitivity Level:** [CRITICAL | HIGH | MEDIUM | LOW | MINIMAL]
**AI Exposure Direction:** [Negative — structural threat | Positive — structural tailwind | Mixed — material headwinds and tailwinds]

### Industry & Exposure Profile

[2–4 sentences. Identify the industry, core revenue streams, and primary AI \
exposure channels. Name specific products or business lines that are exposed. \
Be precise: is the exposure to AI as a competitor, an enabler, an input cost, \
or a demand driver?]

### Sensitivity Drivers

[4–6 bullet points. Each must name a specific mechanism and a concrete \
announcement scenario that would cause a re-rating. Format strictly as:]

- **[Mechanism label]:** [Precise explanation. Which AI announcement? In which \
  direction does the stock move? Why does the market re-price?]

### Headline Risk Scenarios

**Announcements likely to cause a negative stock reaction:**

[2–4 bullets. Specific AI announcements, named organisations where relevant, \
brief causal logic.]

**Announcements likely to cause a positive stock reaction:**

[2–4 bullets. Same format.]

### Structural Context

[2–3 sentences. Place this company in the AI disruption/adoption curve. \
Reference comparable companies that have already experienced AI-driven \
re-ratings where the parallel is genuine and instructive.]

### Verdict

**[CRITICAL | HIGH | MEDIUM | LOW | MINIMAL] sensitivity.** [1–2 sentences \
justifying the rating and stating plainly whether a portfolio manager should \
set AI-announcement alerts for this stock.]
"""

_USER_PROMPT_TEMPLATE = """\
Company: {company}

Claims extracted from the most recent earnings call:
{claims_text}

Produce the AI Announcement Sensitivity section for {company}.\
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def assess_ai_sensitivity(
    company: str,
    claims:  list[Claim],
    client:  Any,
) -> str:
    """Call Claude to produce an AI-announcement sensitivity section.

    Args:
        company: Company ticker (e.g. "BA", "TSLA", "NFLX").
        claims:  Claims extracted from the current-quarter transcript.
        client:  An instantiated ``anthropic.Anthropic()`` client.

    Returns:
        Markdown string for the section (begins with ``## AI Announcement
        Sensitivity``), or a short fallback placeholder if the API call fails.
    """
    # Build a compact claim digest (category + polarity + statement).
    # Deduplicate on the first 80 chars of the claim text; cap at 50 entries
    # to stay comfortably within the model's context window.
    if claims:
        seen: set[str] = set()
        lines: list[str] = []
        for c in claims:
            key = c.claim[:80]
            if key not in seen:
                seen.add(key)
                lines.append(
                    f"- [{c.category.value}] [{c.polarity.value}] {c.claim}"
                )
        claims_text = "\n".join(lines[:50])
    else:
        claims_text = (
            "(No claims extracted — analysis based on company profile only.)"
        )

    user_prompt = _USER_PROMPT_TEMPLATE.format(
        company=company,
        claims_text=claims_text,
    )

    for attempt in (1, 2):
        try:
            response = client.messages.create(
                model="claude-opus-4-6",
                max_tokens=1800,
                temperature=0,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            raw = response.content[0].text.strip()
            logger.info(
                "AI sensitivity assessment generated for %s (%d chars)",
                company, len(raw),
            )
            return raw

        except Exception as exc:
            if attempt == 1:
                logger.warning(
                    "AI sensitivity: API error attempt 1 (%s), retrying…", exc
                )
            else:
                logger.error(
                    "AI sensitivity: API error attempt 2 (%s), skipping", exc
                )

    return (
        "## AI Announcement Sensitivity\n\n"
        "_Assessment unavailable — API error during generation._"
    )
