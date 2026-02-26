"""disruption_lag.py — AI Disruption Lag Analysis.

Identifies which competitive advantages described in an earnings call are already
economically replicable by currently available frontier AI capabilities — and
measures how long that gap has existed without a market repricing.

The core insight: markets systematically lag in repricing companies whose core
business functions have been technologically disrupted.  When the capability has
existed for >12 months and the stock has not yet repriced, that lag is a signal.

"Economic viability date" is strictly defined as: the date on which the AI
capability became cost-competitive with a human equivalent at ≥80% quality and
was deployable at commercial scale.  This is NOT the technical release date —
it is the date of practical, economically-rational deployment.

No LLM calls for the capability database (static, deterministic).
One structured LLM call (temperature=0) for the per-company matching assessment.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict

from llm_extract import Claim, Category

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AI Capability Database
# Each entry is the date the capability became *economically viable*, not the
# date it was technically released.  The distinction is critical for accurate
# lag computation.
# ---------------------------------------------------------------------------

_CAPABILITY_DB: list[dict] = [
    {
        "id": "nlp_text_analysis",
        "name": "Natural Language Understanding & Text Analysis",
        "description": (
            "Automated reading, summarising, classifying and extracting insights "
            "from unstructured text at human parity — reports, emails, contracts, "
            "transcripts, filings.  Viable at scale since GPT-3.5 API availability."
        ),
        "viable_since": date(2022, 11, 30),   # ChatGPT / GPT-3.5 API launch
        "key_enablers": "ChatGPT / GPT-3.5 (Nov 2022)",
        "business_functions": [
            "research", "analysis", "reporting", "content creation", "editorial",
            "compliance review", "document review", "text processing",
            "knowledge management", "information synthesis", "data curation",
        ],
        "precedents": [
            {
                "company": "Chegg", "ticker": "CHGG",
                "peak_drop_pct": -40, "lag_months": 5,
                "description": (
                    "Tutoring/homework-help platform; Q1 2023 earnings guidance cut "
                    "explicitly cited ChatGPT adoption.  Stock peaked Nov 2022."
                ),
            },
            {
                "company": "Content / marketing agencies (generic)", "ticker": None,
                "peak_drop_pct": -35, "lag_months": 12,
                "description": (
                    "Commodity content generation fully commoditised; "
                    "premium editorial partially repriced over 12–18 months."
                ),
            },
        ],
    },
    {
        "id": "code_generation",
        "name": "Code Generation & Software-Development Automation",
        "description": (
            "AI-assisted or semi-autonomous generation of production-quality code, "
            "tests, and documentation — reducing engineering-headcount requirements "
            "for routine development tasks by 30–60% at comparable quality."
        ),
        "viable_since": date(2023, 2, 7),     # GitHub Copilot GA + mass ChatGPT adoption
        "key_enablers": "GitHub Copilot GA (Feb 2023), GPT-4 (Mar 2023)",
        "business_functions": [
            "software development", "engineering", "IT services",
            "outsourced development", "QA testing", "code review",
            "technical documentation", "software maintenance",
        ],
        "precedents": [
            {
                "company": "Upwork", "ticker": "UPWK",
                "peak_drop_pct": -30, "lag_months": 8,
                "description": (
                    "Freelance platform; AI coding tools reduced demand for "
                    "contract developers; repriced over 8 months post-Copilot GA."
                ),
            },
            {
                "company": "IT-staffing / offshore dev firms (generic)", "ticker": None,
                "peak_drop_pct": -20, "lag_months": 18,
                "description": (
                    "Gradual repricing of offshore/nearshore software-development "
                    "services as enterprise Copilot adoption scaled."
                ),
            },
        ],
    },
    {
        "id": "research_synthesis",
        "name": "Research Synthesis & Knowledge Retrieval",
        "description": (
            "Automated synthesis of large volumes of structured and unstructured "
            "data into actionable insights — replicating the core analytical output "
            "of research analysts, consultants, and knowledge workers at GPT-4 quality."
        ),
        "viable_since": date(2023, 3, 14),    # GPT-4 release
        "key_enablers": "GPT-4 (Mar 2023), Claude 2 (Jul 2023)",
        "business_functions": [
            "equity research", "market research", "due diligence",
            "competitive intelligence", "consulting", "advisory",
            "literature review", "analyst coverage", "investment thesis",
            "sector analysis", "primary research", "knowledge services",
        ],
        "precedents": [
            {
                "company": "Chegg", "ticker": "CHGG",
                "peak_drop_pct": -40, "lag_months": 5,
                "description": (
                    "Research/tutoring overlap; GPT-4 surpassed tutoring benchmarks "
                    "within weeks of release.  Chegg CEO publicly cited ChatGPT."
                ),
            },
            {
                "company": "Premium research-subscription services (generic)", "ticker": None,
                "peak_drop_pct": -15, "lag_months": 18,
                "description": (
                    "Niche research aggregators facing substitution pressure; "
                    "repricing gradual as enterprise LLM adoption scaled."
                ),
            },
        ],
    },
    {
        "id": "document_processing",
        "name": "Document Processing, Data Extraction & Classification",
        "description": (
            "Automated extraction of structured data from unstructured documents "
            "(invoices, contracts, forms, filings) at accuracy rates exceeding human "
            "operators — at near-zero marginal cost per document, no retraining required."
        ),
        "viable_since": date(2022, 11, 30),   # GPT-3.5 API
        "key_enablers": "GPT-3.5 API (Nov 2022), Azure Form Recognizer + GPT (2023)",
        "business_functions": [
            "document processing", "back office", "data entry",
            "invoice processing", "contract extraction", "KYC",
            "compliance documentation", "records management",
            "OCR processing", "form processing", "data capture",
        ],
        "precedents": [
            {
                "company": "BPO / document-processing vendors (generic)", "ticker": None,
                "peak_drop_pct": -25, "lag_months": 14,
                "description": (
                    "Offshore document processing commoditised by LLM-based extraction; "
                    "repricing lagged ~14 months behind capability availability."
                ),
            },
        ],
    },
    {
        "id": "customer_service_automation",
        "name": "Customer Service & Conversational AI",
        "description": (
            "End-to-end customer-support automation — resolving complex multi-turn "
            "queries, complaints, and requests without human escalation — at quality "
            "levels acceptable to end users for 60–80% of interaction types."
        ),
        "viable_since": date(2023, 3, 14),    # GPT-4 with system prompts
        "key_enablers": "GPT-4 (Mar 2023), Claude 3 Opus (Mar 2024)",
        "business_functions": [
            "customer service", "customer support", "call centre",
            "contact centre", "help desk", "technical support",
            "customer success", "retention", "live chat",
            "customer experience", "client servicing",
        ],
        "precedents": [
            {
                "company": "Teleperformance", "ticker": "TEP",
                "peak_drop_pct": -45, "lag_months": 4,
                "description": (
                    "Largest BPO globally; -45% in 2023 driven substantially by "
                    "AI customer-service displacement fears.  Fastest recorded lag."
                ),
            },
            {
                "company": "Customer-service BPO firms (generic)", "ticker": None,
                "peak_drop_pct": -25, "lag_months": 16,
                "description": (
                    "Contact-centre outsourcers repriced as enterprise AI-chatbot "
                    "adoption scaled through 2023–2024."
                ),
            },
        ],
    },
    {
        "id": "financial_analysis_modelling",
        "name": "Financial Analysis, Modelling & Scenario Planning",
        "description": (
            "AI-driven financial modelling, variance analysis, scenario planning, "
            "and data visualisation — reducing time-to-insight for structured financial "
            "datasets by 70–90% vs. manual analyst workflows at comparable accuracy."
        ),
        "viable_since": date(2023, 7, 6),     # GPT-4 Code Interpreter / Advanced Data Analysis
        "key_enablers": "GPT-4 Code Interpreter (Jul 2023), Claude 3.5 Sonnet (Jun 2024)",
        "business_functions": [
            "financial modelling", "FP&A", "financial planning", "scenario analysis",
            "valuation", "investment analysis", "credit analysis", "risk modelling",
            "financial reporting", "budget forecasting", "financial analytics",
        ],
        "precedents": [
            {
                "company": "FinTech analytics SaaS vendors (generic)", "ticker": None,
                "peak_drop_pct": -20, "lag_months": 18,
                "description": (
                    "Point solutions for financial analysis facing substitution by "
                    "general-purpose LLMs; repricing gradual as capabilities matured."
                ),
            },
        ],
    },
    {
        "id": "legal_review",
        "name": "Legal Document Review & Contract Analysis",
        "description": (
            "Automated first-pass review, risk-flagging, and summarisation of "
            "legal documents at junior-associate quality for routine review tasks — "
            "contracts, NDAs, regulatory filings, litigation documents."
        ),
        "viable_since": date(2023, 3, 14),    # GPT-4
        "key_enablers": "GPT-4 (Mar 2023), Harvey AI (2023), Claude 2 (Jul 2023)",
        "business_functions": [
            "legal review", "contract review", "contract management",
            "legal compliance", "due diligence legal", "regulatory review",
            "eDiscovery", "legal research", "compliance review",
        ],
        "precedents": [
            {
                "company": "LegalZoom / legal-tech platforms (generic)", "ticker": None,
                "peak_drop_pct": -20, "lag_months": 14,
                "description": (
                    "Routine legal-document generation commoditised; "
                    "premium contract review also facing margin compression."
                ),
            },
        ],
    },
    {
        "id": "translation_localisation",
        "name": "Professional Translation & Localisation",
        "description": (
            "Neural machine translation exceeds human professional translators on "
            "standard benchmarks for 50+ language pairs at near-zero marginal cost — "
            "viable for most commercial documentation and content use cases."
        ),
        "viable_since": date(2022, 11, 30),   # GPT-3.5 + DeepL Pro
        "key_enablers": "GPT-3.5 (Nov 2022), DeepL Pro Neural (2022)",
        "business_functions": [
            "translation", "localisation", "language services",
            "multilingual content", "international documentation", "subtitling",
            "language operations",
        ],
        "precedents": [
            {
                "company": "Human translation agencies (generic)", "ticker": None,
                "peak_drop_pct": -40, "lag_months": 24,
                "description": (
                    "Professional translation volumes declining; "
                    "price compression severe; slowest repricing due to fragmentation."
                ),
            },
        ],
    },
    {
        "id": "multimodal_analysis",
        "name": "Multimodal Document & Image Analysis",
        "description": (
            "AI systems that combine vision and text understanding — enabling "
            "automated analysis of charts, diagrams, scanned documents, and images "
            "at human-expert accuracy for standard inspection and extraction tasks."
        ),
        "viable_since": date(2023, 11, 6),    # GPT-4V general availability
        "key_enablers": "GPT-4V GA (Nov 2023), Gemini 1.5 Pro (Feb 2024), Claude 3 (Mar 2024)",
        "business_functions": [
            "visual inspection", "quality control", "image analysis",
            "document digitisation", "chart analysis", "medical imaging review",
            "satellite imagery analysis", "visual data extraction",
        ],
        "precedents": [
            {
                "company": "Specialised visual-AI point vendors (generic)", "ticker": None,
                "peak_drop_pct": -20, "lag_months": 14,
                "description": (
                    "Point-solution visual-AI vendors repriced as GPT-4V-class "
                    "APIs commoditised their core capability."
                ),
            },
        ],
    },
    {
        "id": "agentic_workflow_automation",
        "name": "Agentic Workflow Automation (Multi-Step AI Agents)",
        "description": (
            "AI agents that autonomously execute multi-step workflows — browsing, "
            "writing, coding, API calls, form-filling — without human handholding. "
            "Replaces entire knowledge-worker task sequences, not just individual steps."
        ),
        "viable_since": date(2024, 6, 20),    # Claude 3.5 Sonnet / GPT-4o tools at scale
        "key_enablers": "Claude 3.5 Sonnet (Jun 2024), GPT-4o with tools, OpenAI Swarm (Oct 2024)",
        "business_functions": [
            "workflow automation", "process automation", "RPA",
            "business process outsourcing", "administrative workflows",
            "research workflows", "operations automation",
            "knowledge work automation", "multi-step process execution",
        ],
        "precedents": [
            {
                "company": "RPA vendors / BPO knowledge work (generic)", "ticker": None,
                "peak_drop_pct": None, "lag_months": 8,
                "description": (
                    "Very recent capability — market has not fully priced agentic "
                    "displacement of knowledge-worker workflows.  "
                    "Highest unpriced lag potential heading into 2025–2026."
                ),
            },
        ],
    },
    {
        "id": "complex_reasoning",
        "name": "Complex Reasoning, Strategic Analysis & Expert Problem-Solving",
        "description": (
            "O1-class models capable of extended multi-step reasoning — approaching "
            "expert-level performance on STEM, financial, strategic, and analytical "
            "tasks that previously required senior human expertise."
        ),
        "viable_since": date(2024, 9, 12),    # OpenAI o1 release
        "key_enablers": "OpenAI o1 (Sep 2024), o1-pro, Claude 3.7 Sonnet (Feb 2025)",
        "business_functions": [
            "strategic consulting", "expert advisory", "scientific research",
            "complex analysis", "senior consulting", "technical architecture",
            "investment strategy formulation", "advanced risk assessment",
            "expert systems", "high-complexity problem solving",
        ],
        "precedents": [
            {
                "company": "Management-consulting firms / expert advisory (generic)", "ticker": None,
                "peak_drop_pct": None, "lag_months": 6,
                "description": (
                    "Very recent — junior-to-mid consulting workflows at risk; "
                    "market has not yet priced this disruption.  "
                    "Estimated repricing lag: 6–18 months from analysis date."
                ),
            },
        ],
    },
]

# O(1) lookup by capability id
_CAPABILITY_BY_ID: dict[str, dict] = {c["id"]: c for c in _CAPABILITY_DB}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class DisruptionSignal(BaseModel):
    """One matched disruption risk: a moat claim vs. an AI capability."""

    model_config = ConfigDict(extra="forbid")

    moat_claim:                str
    capability_id:             str
    capability_name:           str
    viable_since:              str    # ISO date string
    lag_months:                int
    lag_label:                 str    # "CRITICAL" | "HIGH" | "MEDIUM" | "LOW"
    replaceability_score:      float  # 0.0–1.0
    replaceability_reasoning:  str
    management_awareness_note: str
    best_analogue:             Optional[str] = None   # e.g. "Chegg -40% (5mo lag)"


class DisruptionLagResult(BaseModel):
    """Aggregate disruption lag assessment for one company."""

    model_config = ConfigDict(extra="forbid")

    overall_score:          str    # "CRITICAL" | "HIGH" | "MEDIUM" | "LOW" | "MINIMAL"
    n_signals:              int
    management_awareness:   str    # "low" | "medium" | "high"
    signals:                list[DisruptionSignal]
    analyst_narrative:      str    # 2–3 sentence synthesis for a portfolio manager


# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a senior technology equity analyst specialising in AI disruption and \
competitive moat erosion. Your task: assess which competitive advantages described \
in an earnings call are already economically replicable by currently available \
frontier AI capabilities.

STRICT RULES — read carefully before producing output:

1. ECONOMIC VIABILITY ONLY.  Only flag capabilities that ALREADY EXIST and are \
   ECONOMICALLY VIABLE — meaning: cost per task is below the human equivalent at \
   ≥80% quality, deployable at commercial scale TODAY.  Do NOT extrapolate to \
   future capabilities.

2. GENUINE MOAT MATCH ONLY.  A moat based on proprietary DATA or NETWORK EFFECTS \
   or PHYSICAL ASSETS or REGULATORY LICENCES is generally NOT AI-replicable — do \
   not flag these.  Only flag moats based on HUMAN COGNITIVE LABOUR that AI can now \
   perform.

3. PARTIAL VS. FULL.  Score replaceability_score carefully:
   - 1.0 = AI fully replicates this function at parity cost and quality
   - 0.7–0.9 = AI replicates the core output; human oversight still adds value
   - 0.4–0.6 = AI automates the routine portion; expert judgement still needed
   - < 0.4 = AI is an efficiency tool, not a moat-eroder; do NOT include

4. MANAGEMENT AWARENESS.  If management explicitly describes AI adoption or \
   transformation of the flagged function, note this — it reduces lag severity.

5. COMPETITIVE SYMMETRY.  If the entire sector faces the same disruption, \
   individual repricing may not occur — note this in your reasoning.

6. FALSE POSITIVES ARE WORSE THAN FALSE NEGATIVES.  When in doubt, do not flag.

7. OUTPUT FORMAT.  Return ONLY valid JSON matching the schema below. \
   No preamble, no markdown fences, no text outside the JSON structure.\
"""

_USER_TEMPLATE = """\
Company: {company}
Analysis date: {analysis_date}

MOAT-RELATED CLAIMS FROM MOST RECENT EARNINGS CALL ({n_claims} claims):
{moat_claims_text}

AVAILABLE AI CAPABILITIES (economically viable as of {analysis_date}):
{capabilities_text}

For each claim that has a genuine capability match (replaceability_score ≥ 0.4), \
produce one signal entry.  If a claim has no genuine match, exclude it.
If ALL claims are based on non-replicable moats (data assets, network effects, \
physical assets, regulatory barriers), return an empty signals list with \
overall_score "MINIMAL".

Return JSON matching this exact schema — no other text:
{{
  "overall_score": "<CRITICAL|HIGH|MEDIUM|LOW|MINIMAL>",
  "management_awareness": "<low|medium|high>",
  "analyst_narrative": "<2–3 sentences for a portfolio manager: what is the core \
disruption risk, how large is the exposed revenue base, and what is the key \
signal to monitor?>",
  "signals": [
    {{
      "moat_claim": "<exact or near-exact claim text from input>",
      "capability_id": "<id string from the capability list above>",
      "replaceability_score": <float 0.0–1.0>,
      "replaceability_reasoning": "<1–2 sentences: what precisely does the AI do \
that replicates this function? Be specific about mechanism, not generic.>",
      "management_awareness_note": "<evidence that management is / is not aware \
of this AI risk to the specific function; 'None evident' if no mention>"
    }}
  ]
}}\
"""


# ---------------------------------------------------------------------------
# Moat-claim identification (deterministic pre-filter)
# ---------------------------------------------------------------------------

# Categories most likely to contain moat / competitive-advantage claims
_MOAT_CATEGORIES: frozenset[str] = frozenset({
    Category.competition.value,
    Category.ops_execution.value,
    Category.guidance.value,
    Category.pricing_margin.value,
    Category.demand.value,
})

# Keywords that signal a competitive-advantage or human-expertise moat
_MOAT_KEYWORDS: tuple[str, ...] = (
    "proprietary", "unique", "differentiated", "irreplaceable", "advantage",
    "barrier", "exclusive", "platform", "ecosystem", "network effect",
    "expertise", "specialized", "specialist", "best-in-class", "leading",
    "market-leading", "core competency", "competitive", "our strength",
    "our team", "our people", "our technology", "our platform",
    "our data", "our relationships", "our process", "our methodology",
    "our model", "our system", "our approach", "unmatched", "superior",
    "human capital", "talent", "knowledge", "experience", "reputation",
)


def identify_moat_claims(claims: list[Claim], max_claims: int = 25) -> list[Claim]:
    """Return the subset of claims most likely to describe competitive moats.

    Primary filter: category is in ``_MOAT_CATEGORIES`` AND polarity is
    positive or neutral AND at least one moat keyword is present.

    Fallback (if primary filter yields nothing): all positive claims from
    moat categories, up to *max_claims*.
    """
    primary: list[Claim] = []
    for c in claims:
        if c.category.value not in _MOAT_CATEGORIES:
            continue
        if c.polarity.value not in ("positive", "neutral"):
            continue
        c_lower = c.claim.lower()
        if any(kw in c_lower for kw in _MOAT_KEYWORDS):
            primary.append(c)

    if primary:
        return primary[:max_claims]

    # Fallback: positive claims from moat categories, no keyword filter
    fallback = [
        c for c in claims
        if c.category.value in _MOAT_CATEGORIES
        and c.polarity.value == "positive"
    ]
    return fallback[:max_claims]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_lag_months(viable_since: date, today: date) -> int:
    """Return whole months between viable_since and today (floor)."""
    return max(0, (today - viable_since).days // 30)


def _lag_label(lag_months: int) -> str:
    """Classify lag severity by how long the capability has existed."""
    if lag_months >= 18:
        return "CRITICAL"
    if lag_months >= 12:
        return "HIGH"
    if lag_months >= 6:
        return "MEDIUM"
    return "LOW"   # 0–5 months: capability very recent


def _best_analogue_str(cap: dict) -> Optional[str]:
    """Return the most concrete historical precedent string, or None."""
    for p in cap.get("precedents", []):
        pct = p.get("peak_drop_pct")
        lag = p.get("lag_months")
        if pct and lag:
            company = p["company"]
            return f"{company}: {pct:+d}% ({lag}mo lag)"
    return None


def _format_moat_claims(claims: list[Claim]) -> str:
    lines = []
    for i, c in enumerate(claims, 1):
        lines.append(f"{i}. [{c.category.value}] [{c.polarity.value}] {c.claim}")
    return "\n".join(lines)


def _format_capabilities(caps: list[dict]) -> str:
    lines = []
    for cap in caps:
        vs = cap["viable_since"].strftime("%b %Y")
        lines.append(
            f"- [{cap['id']}] {cap['name']} (viable since {vs}): {cap['description']}"
        )
    return "\n".join(lines)


def _call_llm(client: Any, user_prompt: str) -> Optional[str]:
    """Single LLM call with one retry. Returns raw text or None."""
    for attempt in (1, 2):
        try:
            response = client.messages.create(
                model="claude-opus-4-6",
                max_tokens=2400,
                temperature=0,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return response.content[0].text
        except Exception as exc:
            if attempt == 1:
                logger.warning("Disruption lag: API error attempt 1 (%s), retrying…", exc)
            else:
                logger.error("Disruption lag: API error attempt 2 (%s), skipping", exc)
    return None


def _parse_llm_response(
    raw:  str,
    today: date,
) -> Optional[DisruptionLagResult]:
    """Parse the structured JSON from the LLM and build a DisruptionLagResult."""
    text = raw.strip()
    if text.startswith("```"):
        text = "\n".join(
            ln for ln in text.splitlines() if not ln.startswith("```")
        ).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.error("Disruption lag: JSON parse error — %s | raw: %.200s", exc, raw)
        return None

    # overall_score from LLM is used only as a fallback; we recompute it
    # deterministically below from the validated signals to guarantee
    # self-consistency (LLM could assert "CRITICAL" and then all signals
    # get dropped, or assert "LOW" while signals compute to 28-month lags).
    management_awareness = str(data.get("management_awareness", "medium")).lower()
    analyst_narrative    = str(data.get("analyst_narrative", ""))
    raw_signals = data.get("signals", [])
    if not isinstance(raw_signals, list):
        logger.warning("Disruption lag: 'signals' field is not a list (%r) — defaulting to []",
                       type(raw_signals).__name__)
        raw_signals = []

    if management_awareness not in {"low", "medium", "high"}:
        management_awareness = "medium"

    signals: list[DisruptionSignal] = []
    for s in raw_signals:
        cap_id = s.get("capability_id", "")
        cap    = _CAPABILITY_BY_ID.get(cap_id)
        if cap is None:
            logger.warning("Disruption lag: unknown capability_id '%s' — skipping", cap_id)
            continue

        moat_claim = s.get("moat_claim", "").strip()
        if not moat_claim:
            logger.warning(
                "Disruption lag: signal with capability '%s' has empty moat_claim — skipping",
                cap_id,
            )
            continue

        try:
            rep_score = max(0.0, min(1.0, float(s.get("replaceability_score", 0.5))))
        except (TypeError, ValueError):
            rep_score = 0.5

        lag_months = _compute_lag_months(cap["viable_since"], today)

        signals.append(DisruptionSignal(
            moat_claim                = moat_claim,
            capability_id             = cap_id,
            capability_name           = cap["name"],
            viable_since              = cap["viable_since"].isoformat(),
            lag_months                = lag_months,
            lag_label                 = _lag_label(lag_months),
            replaceability_score      = round(rep_score, 2),
            replaceability_reasoning  = s.get("replaceability_reasoning", ""),
            management_awareness_note = s.get("management_awareness_note", ""),
            best_analogue             = _best_analogue_str(cap),
        ))

    # Sort: longest lag first, then highest replaceability
    signals.sort(key=lambda sig: (-sig.lag_months, -sig.replaceability_score))

    # Derive overall_score deterministically from validated signals so it is
    # always consistent with what the table shows.  LLM assertion is discarded.
    _LABEL_RANK = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
    _RANK_LABEL = {4: "CRITICAL", 3: "HIGH", 2: "MEDIUM", 1: "LOW", 0: "MINIMAL"}
    if signals:
        worst_rank = max(_LABEL_RANK.get(s.lag_label, 0) for s in signals)
        overall_score = _RANK_LABEL[worst_rank]
    else:
        overall_score = "MINIMAL"

    return DisruptionLagResult(
        overall_score        = overall_score,
        n_signals            = len(signals),
        management_awareness = management_awareness,
        signals              = signals,
        analyst_narrative    = analyst_narrative,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_disruption_lag(
    company: str,
    claims:  list[Claim],
    client:  Any,
) -> Optional[DisruptionLagResult]:
    """Compute AI disruption lag signals for a company.

    Identifies moat-related claims from the extracted earnings-call claims,
    matches them against the AI capability database, and returns a structured
    assessment of which competitive advantages are already economically
    replicable — and for how long.

    Args:
        company: Company ticker (e.g. "BA").
        claims:  Claims extracted from the current-quarter transcript.
        client:  An instantiated ``anthropic.Anthropic()`` client.

    Returns:
        :class:`DisruptionLagResult` on success, or ``None`` if no moat
        claims are found or if the LLM call fails.
    """
    today = date.today()

    # 1. Identify moat claims
    moat_claims = identify_moat_claims(claims)
    if not moat_claims:
        logger.info("No moat claims identified for %s — disruption lag skipped", company)
        return None

    # 2. All capabilities viable as of today
    viable_caps = [c for c in _CAPABILITY_DB if c["viable_since"] <= today]

    # 3. Build prompt
    # Escape any literal braces in transcript-derived text so str.format() does
    # not misinterpret them as named placeholders and raise a KeyError.
    safe_moat_claims = _format_moat_claims(moat_claims).replace("{", "{{").replace("}", "}}")
    user_prompt = _USER_TEMPLATE.format(
        company           = company,
        analysis_date     = today.strftime("%B %d, %Y"),
        n_claims          = len(moat_claims),
        moat_claims_text  = safe_moat_claims,
        capabilities_text = _format_capabilities(viable_caps),
    )

    # 4. LLM call
    raw = _call_llm(client, user_prompt)
    if raw is None:
        return None

    # 5. Parse and enrich
    result = _parse_llm_response(raw, today)
    if result:
        logger.info(
            "Disruption lag %s: score=%s  signals=%d  mgmt_awareness=%s",
            company, result.overall_score, result.n_signals, result.management_awareness,
        )
    return result
