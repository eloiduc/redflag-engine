"""prediction_markets.py — Prediction market context for earnings analysis.

Fetches active markets from Polymarket (public Gamma API, no auth required)
and optionally Kalshi (REST API — set KALSHI_API_KEY env var to enable), then
cross-references them against the claims extracted from the current-quarter
transcript to surface confirmations and contradictions.

The intelligence layer:
  - Earnings transcripts represent management's narrative.
  - Prediction markets represent aggregated public belief with real money.
  - When the two diverge materially, that gap is where scrutiny is warranted.

No LLM calls — fully deterministic.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict

from llm_extract import Claim, Polarity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_POLYMARKET_BASE  = "https://gamma-api.polymarket.com"
_KALSHI_BASE      = "https://trading-api.kalshi.com/trade-api/v2"
_MIN_VOLUME_USD   = 5_000    # Discard illiquid / dormant markets
_MAX_MARKETS      = 12       # Cap total returned markets
_STRONG_PROB_LOW  = 0.35     # Below → strong bearish signal (market leans No)
_STRONG_PROB_HIGH = 0.65     # Above → strong bullish signal (market leans Yes)
_RELEVANCE_MIN    = 0.03     # Minimum Jaccard overlap for claim cross-reference

# Ticker → canonical company name for search query building
_COMPANY_NAMES: dict[str, str] = {
    "BA":   "Boeing",
    "TSLA": "Tesla",
    "NFLX": "Netflix",
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Google Alphabet",
    "AMZN": "Amazon",
    "META": "Meta Platforms",
    "NVDA": "Nvidia",
    "JPM":  "JPMorgan",
    "GS":   "Goldman Sachs",
    "MS":   "Morgan Stanley",
    "RTX":  "Raytheon",
    "LMT":  "Lockheed Martin",
    "NOC":  "Northrop Grumman",
    "SPR":  "Spirit AeroSystems",
    "TDG":  "TransDigm",
    "UBER": "Uber",
    "LYFT": "Lyft",
    "ABNB": "Airbnb",
}

# Positive-framing words: a Yes outcome is directionally good for the company.
# Negation-qualified verbs (avoid / prevent / survive / withstand / overcome)
# are included because market questions like "Will X avoid bankruptcy?" have
# Yes = company escapes bad outcome = bullish.  Without these, the negative
# noun ("bankruptcy") would dominate and produce the wrong framing signal.
_POS_FRAME: frozenset[str] = frozenset({
    "reach", "achieve", "beat", "exceed", "above", "grow", "gain", "approve",
    "certify", "complete", "deliver", "pass", "win", "launch", "secure",
    "maintain", "increase", "expand", "recover", "profitable", "profit",
    "success", "upgrade", "raise", "positive", "growth", "resume",
    # Negation-qualified verbs: Yes = company avoids / escapes a negative outcome
    "avoid", "prevent", "survive", "withstand", "overcome", "resolve",
})

# Negative-framing words: a Yes outcome is directionally bad for the company
_NEG_FRAME: frozenset[str] = frozenset({
    "miss", "fail", "below", "decline", "fall", "lose", "cut", "reject",
    "deny", "cancel", "delay", "default", "drop", "reduce", "exit", "close",
    "bankrupt", "bankruptcy", "downgrade", "layoff", "recall", "investigation",
    "fine", "penalty", "lawsuit", "violation", "loss", "losses", "negative",
    "crash", "halt", "suspend", "delist", "probe", "charge",
})

_STOP_WORDS: frozenset[str] = frozenset({
    "the", "a", "an", "is", "will", "be", "to", "of", "in", "for",
    "and", "or", "by", "on", "at", "from", "with", "its", "their",
    "this", "that", "than", "are", "was", "were", "has", "have",
    "had", "not", "no", "do", "does", "did", "can", "could", "would",
    "should", "may", "might", "must", "shall", "if", "as", "but",
    "which", "who", "what", "when", "where", "how", "why", "any",
    "all", "each", "more", "most", "than", "such", "also", "into",
    "per", "over", "under", "about", "after", "before", "during",
})


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class PredictionMarket(BaseModel):
    """A single prediction market fetched from Polymarket or Kalshi."""

    model_config = ConfigDict(extra="forbid")

    platform:        str
    question:        str
    yes_probability: float           # 0.0 – 1.0
    volume_usd:      float
    liquidity_usd:   float  = 0.0
    expires:         Optional[str] = None   # ISO date string
    url:             Optional[str] = None
    relevance_score: float  = 0.0    # computed post-retrieval


class MarketClaimCrossRef(BaseModel):
    """A cross-reference between one prediction market and one extracted claim."""

    model_config = ConfigDict(extra="forbid")

    market_question:  str
    platform:         str
    yes_probability:  float
    volume_usd:       float
    expires:          Optional[str] = None
    url:              Optional[str] = None
    claim_text:       str
    claim_polarity:   str   # Polarity enum value
    claim_category:   str   # Category enum value, human-readable
    alignment:        str   # "CONTRADICTS" | "CONFIRMS" | "NEUTRAL"
    interpretation:   str


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

def _get_json(url: str, headers: dict[str, str] | None = None, timeout: int = 12) -> Any:
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "RedFlagEngine/1.0")
    req.add_header("Accept",     "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# Platform clients
# ---------------------------------------------------------------------------

class _PolymarketClient:
    """Polymarket Gamma public API — no authentication required."""

    def search(self, query: str, limit: int = 30) -> list[PredictionMarket]:
        params = urllib.parse.urlencode({
            "search": query,
            "active": "true",
            "closed": "false",
            "limit":  str(limit),
        })
        url = f"{_POLYMARKET_BASE}/markets?{params}"
        try:
            data = _get_json(url)
        except Exception as exc:
            logger.debug("Polymarket search '%s' failed: %s", query, exc)
            return []

        results: list[PredictionMarket] = []
        for item in (data if isinstance(data, list) else []):
            try:
                question = item.get("question", "").strip()
                if not question:
                    continue

                # outcomePrices is a JSON string e.g. '["0.65","0.35"]'
                prices_raw = item.get("outcomePrices", "[]")
                prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
                try:
                    yes_prob = max(0.0, min(1.0, float(prices[0]))) if prices else 0.5
                except (TypeError, ValueError, IndexError):
                    yes_prob = 0.5
                try:
                    volume = float(item.get("volume", 0) or 0)
                except (TypeError, ValueError):
                    volume = 0.0
                try:
                    liquidity = float(item.get("liquidity", 0) or 0)
                except (TypeError, ValueError):
                    liquidity = 0.0
                end_date  = (item.get("endDate") or "")[:10] or None
                slug      = item.get("slug", "")
                url_str   = f"https://polymarket.com/event/{slug}" if slug else None

                if volume < _MIN_VOLUME_USD:
                    continue

                results.append(PredictionMarket(
                    platform        = "Polymarket",
                    question        = question,
                    yes_probability = round(yes_prob, 3),
                    volume_usd      = round(volume,   0),
                    liquidity_usd   = round(liquidity, 0),
                    expires         = end_date,
                    url             = url_str,
                ))
            except Exception:
                continue
        return results


class _KalshiClient:
    """Kalshi REST API — requires KALSHI_API_KEY environment variable."""

    def __init__(self) -> None:
        self._api_key = os.environ.get("KALSHI_API_KEY", "").strip()

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    def search(self, query: str, limit: int = 15) -> list[PredictionMarket]:
        if not self.available:
            return []
        params = urllib.parse.urlencode({
            "search": query,
            "status": "open",
            "limit":  str(limit),
        })
        url = f"{_KALSHI_BASE}/markets?{params}"
        try:
            data = _get_json(url, headers={"Authorization": f"Bearer {self._api_key}"})
        except Exception as exc:
            logger.debug("Kalshi search '%s' failed: %s", query, exc)
            return []

        results: list[PredictionMarket] = []
        for item in (data.get("markets", []) if isinstance(data, dict) else []):
            try:
                title = item.get("title", "").strip()
                if not title:
                    continue
                # Kalshi v2 prices are integers 0–100 (cents)
                try:
                    yes_ask = float(item.get("yes_ask", 50) or 50)
                    yes_bid = float(item.get("yes_bid", 50) or 50)
                    yes_prob = max(0.0, min(1.0, (yes_ask + yes_bid) / 2 / 100.0))
                except (TypeError, ValueError):
                    yes_prob = 0.5
                try:
                    volume = float(item.get("volume", 0) or 0) / 100  # cents → USD
                except (TypeError, ValueError):
                    volume = 0.0
                close    = (item.get("close_time") or "")[:10] or None
                ticker   = item.get("ticker", "")
                url_str  = f"https://kalshi.com/markets/{ticker}" if ticker else None

                if volume < _MIN_VOLUME_USD:
                    continue

                results.append(PredictionMarket(
                    platform        = "Kalshi",
                    question        = title,
                    yes_probability = round(yes_prob, 3),
                    volume_usd      = round(volume, 0),
                    expires         = close,
                    url             = url_str,
                ))
            except Exception:
                continue
        return results


# ---------------------------------------------------------------------------
# Relevance & cross-reference logic
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> frozenset[str]:
    tokens = re.findall(r"[a-z]+", text.lower())
    return frozenset(t for t in tokens if t not in _STOP_WORDS and len(t) > 2)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _infer_framing(question: str) -> str:
    """Infer whether a Yes outcome is directionally positive or negative."""
    q_tokens  = _tokenize(question)
    pos_score = len(q_tokens & _POS_FRAME)
    neg_score = len(q_tokens & _NEG_FRAME)
    if pos_score > neg_score:
        return "positive"
    if neg_score > pos_score:
        return "negative"
    return "neutral"


def _compute_alignment(
    framing:         str,
    yes_probability: float,
    claim_polarity:  Polarity,
) -> tuple[str, str]:
    """Return (alignment_label, interpretation_text).

    Alignment logic:
      positive-framed market + high prob  → bullish signal
      positive-framed market + low prob   → bearish signal
      negative-framed market + high prob  → bearish signal
      negative-framed market + low prob   → bullish signal

    Cross-reference with claim polarity → CONFIRMS / CONTRADICTS / NEUTRAL.
    """
    prob_pct = f"{yes_probability:.0%}"

    # Translate market framing + probability into a directional stance
    if framing == "positive":
        if yes_probability >= _STRONG_PROB_HIGH:
            mkt_stance = "bullish"
            mkt_desc   = f"market strongly expects this positive outcome ({prob_pct})"
        elif yes_probability <= _STRONG_PROB_LOW:
            mkt_stance = "bearish"
            mkt_desc   = f"market is skeptical of this positive outcome ({prob_pct})"
        else:
            return "NEUTRAL", f"Market probability ({prob_pct}) is in the inconclusive range."

    elif framing == "negative":
        if yes_probability >= _STRONG_PROB_HIGH:
            mkt_stance = "bearish"
            mkt_desc   = f"market strongly expects this negative outcome ({prob_pct})"
        elif yes_probability <= _STRONG_PROB_LOW:
            mkt_stance = "bullish"
            mkt_desc   = f"market is skeptical of this negative outcome ({prob_pct})"
        else:
            return "NEUTRAL", f"Market probability ({prob_pct}) is in the inconclusive range."

    else:
        return "NEUTRAL", f"Market probability: {prob_pct}. Framing is ambiguous — review manually."

    # Map claim polarity to a directional stance
    claim_stance = {
        Polarity.positive: "bullish",
        Polarity.negative: "bearish",
        Polarity.neutral:  "neutral",
        Polarity.mixed:    "neutral",
    }.get(claim_polarity, "neutral")

    if claim_stance == "neutral":
        return "NEUTRAL", f"Market: {mkt_desc}. Claim polarity is neutral/mixed."

    if mkt_stance == claim_stance:
        return (
            "CONFIRMS",
            f"Market {mkt_desc}, consistent with management's "
            f"{claim_polarity.value} guidance.",
        )
    return (
        "CONTRADICTS",
        f"Market {mkt_desc}, contradicting management's "
        f"{claim_polarity.value} guidance. Independent verification warranted.",
    )


def _build_queries(company: str, claims: list[Claim]) -> list[str]:
    """Build targeted search queries: company name + category enrichments."""
    name = _COMPANY_NAMES.get(company.upper(), company)
    # Start with ticker itself as first query, then full name
    queries: list[str] = [company.upper(), name]

    category_enrichments = {
        "reg_legal":           f"{name} regulatory investigation",
        "guidance":            f"{name} earnings",
        "liquidity":           f"{name} debt financing",
        "costs_restructuring": f"{name} restructuring layoffs",
        "competition":         f"{name} market share",
    }
    active_cats = {c.category.value for c in claims}
    for cat, qry in category_enrichments.items():
        if cat in active_cats:
            queries.append(qry)

    # Deduplicate, preserve order
    seen: set[str] = set()
    unique: list[str] = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            unique.append(q)

    return unique[:5]  # 5 queries max


def _filter_and_rank(
    markets:  list[PredictionMarket],
    company:  str,
    claims:   list[Claim],
) -> list[PredictionMarket]:
    """Score markets for company relevance; return top _MAX_MARKETS."""
    name           = _COMPANY_NAMES.get(company.upper(), company)
    company_tokens = _tokenize(name) | _tokenize(company)

    claim_tokens: frozenset[str] = frozenset()
    for c in claims:
        claim_tokens = claim_tokens | _tokenize(c.claim)

    scored: list[PredictionMarket] = []
    seen:   set[str]               = set()

    for m in markets:
        q_norm = m.question.strip().lower()
        if q_norm in seen:
            continue
        seen.add(q_norm)

        mq_tokens = _tokenize(m.question)

        # Hard filter: market question must mention the company ticker or primary name
        company_str  = company.lower()
        q_lower      = m.question.lower()
        name_parts   = name.lower().split()      # all words of the company name
        if (
            company_str not in q_lower
            and not any(part in q_lower for part in name_parts if len(part) > 3)
        ):
            continue

        company_overlap = _jaccard(mq_tokens, company_tokens)
        claim_overlap   = _jaccard(mq_tokens, claim_tokens)

        # Relevance: company overlap is the primary signal, claims enrich it
        relevance = min(1.0, company_overlap * 1.8 + claim_overlap * 0.5)
        m = m.model_copy(update={"relevance_score": round(relevance, 3)})
        scored.append(m)

    # Sort by relevance × log(volume): rewarding both topicality and liquidity
    scored.sort(key=lambda m: -(m.relevance_score * math.log1p(m.volume_usd)))
    return scored[:_MAX_MARKETS]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_relevant_markets(
    company:    str,
    claims:     list[Claim],
    now_period: str = "",
) -> list[PredictionMarket]:
    """Fetch and rank prediction markets relevant to this company and its claims.

    Queries Polymarket (always) and Kalshi (when KALSHI_API_KEY is set).
    Results are filtered for minimum liquidity ($5k volume) and ranked by
    relevance × log(volume).

    Args:
        company:    Ticker (e.g. "BA").
        claims:     Claims from the current-quarter transcript.
        now_period: Period label — informational only.

    Returns:
        Up to 12 :class:`PredictionMarket` objects. Empty list on total failure.
    """
    polymarket = _PolymarketClient()
    kalshi     = _KalshiClient()

    queries     = _build_queries(company, claims)
    all_markets: list[PredictionMarket] = []

    for query in queries:
        all_markets.extend(polymarket.search(query))
        if kalshi.available:
            all_markets.extend(kalshi.search(query))

    if not all_markets:
        logger.info("No prediction markets found for %s", company)
        return []

    filtered = _filter_and_rank(all_markets, company, claims)
    logger.info(
        "Prediction markets %s: %d raw → %d after relevance filter",
        company, len(all_markets), len(filtered),
    )
    return filtered


def cross_reference_with_claims(
    markets: list[PredictionMarket],
    claims:  list[Claim],
) -> list[MarketClaimCrossRef]:
    """Cross-reference prediction markets against extracted management claims.

    Only processes markets with a strong directional signal (probability
    outside the 35–65% inconclusive zone).  For each such market, finds the
    most semantically overlapping claim and classifies the relationship as
    CONTRADICTS, CONFIRMS, or NEUTRAL.

    CONTRADICTS entries appear first — they are the most actionable signal.

    Args:
        markets: Output of :func:`find_relevant_markets`.
        claims:  Claims from the current-quarter transcript.

    Returns:
        List of :class:`MarketClaimCrossRef`. Empty if no meaningful pairings.
    """
    if not markets or not claims:
        return []

    refs: list[MarketClaimCrossRef] = []

    for market in markets:
        # Only strong probability signals are worth cross-referencing
        if _STRONG_PROB_LOW < market.yes_probability < _STRONG_PROB_HIGH:
            continue

        mq_tokens = _tokenize(market.question)
        framing   = _infer_framing(market.question)

        # Find the claim with highest keyword overlap
        best_claim: Optional[Claim] = None
        best_score: float           = 0.0
        for claim in claims:
            sc = _jaccard(mq_tokens, _tokenize(claim.claim))
            if sc > best_score:
                best_score = sc
                best_claim = claim

        if best_claim is None or best_score < _RELEVANCE_MIN:
            continue

        alignment, interp = _compute_alignment(
            framing, market.yes_probability, best_claim.polarity
        )

        refs.append(MarketClaimCrossRef(
            market_question  = market.question,
            platform         = market.platform,
            yes_probability  = market.yes_probability,
            volume_usd       = market.volume_usd,
            expires          = market.expires,
            url              = market.url,
            claim_text       = best_claim.claim,
            claim_polarity   = best_claim.polarity.value,
            claim_category   = best_claim.category.value.replace("_", " ").title(),
            alignment        = alignment,
            interpretation   = interp,
        ))

    _order = {"CONTRADICTS": 0, "CONFIRMS": 1, "NEUTRAL": 2}
    refs.sort(key=lambda r: (_order.get(r.alignment, 3), -r.volume_usd))

    n_contra  = sum(1 for r in refs if r.alignment == "CONTRADICTS")
    n_confirm = sum(1 for r in refs if r.alignment == "CONFIRMS")
    logger.info(
        "Cross-refs: %d total  CONTRADICTS=%d  CONFIRMS=%d  NEUTRAL=%d",
        len(refs), n_contra, n_confirm, len(refs) - n_contra - n_confirm,
    )
    return refs
