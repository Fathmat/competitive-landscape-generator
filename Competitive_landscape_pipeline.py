#!/usr/bin/env python3
"""Competitive Landscape Generator (CLG)
-------------------------------------

Goal
----
Given a target company (name + description + optional URL/SIC), this script:

1) Builds a structured TARGET BUSINESS PROFILE using an LLM.
2) Generates COMPETITIVE CANDIDATES via role-specific LLM prompts:
      - direct_competitor
      - adjacent_competitor
      - substitute
      - supplier
3) Resolves candidates to tickers via Financial Modeling Prep (FMP) and enriches
   them with sector/industry/description.
4) Computes a COMPETITIVE PROXIMITY SCORE using:
      - TF-IDF cosine similarity between descriptions
      - Industry similarity heuristic
5) Uses an LLM to classify STRATEGIC RELATIONSHIP between target and each candidate:
      - direct_competitor
      - adjacent_competitor
      - substitute
      - supplier
      - downstream_customer
      - unrelated
6) Exports a grouped competitive landscape.

Design Notes
------------
- Candidate generation is split into multiple prompts (per role) to reduce JSON failures
  and improve role coverage (esp. substitutes and suppliers).
- Robust JSON parsing includes:
    * strict parse
    * best-effort extraction of JSON from surrounding text
    * optional JSON repair round
- Pydantic v1/v2 compatible serialization.

Requirements
------------
- Python 3.9+
- pip install:
    requests
    pandas
    scikit-learn
    openai
    pydantic            (optional but recommended)
    python-dotenv       (optional)
    pyarrow             (optional, for Parquet export)

Environment
-----------
- FMP_API_KEY=<your FMP key>
- OPENAI_API_KEY=<your OpenAI key>

Optional:
- OPENAI_MODEL_EXTRACTION=gpt-5.1
- OPENAI_MODEL_CANDIDATES=gpt-5.1
- OPENAI_MODEL_VALIDATION=gpt-5.1

Usage
-----
python Competitive_landscape_pipeline.py \
  --name "Shopify Inc." \
  --desc "..." \
  --sic "..." \
  --outdir ./runs/shopify_test

"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------
# Optional Pydantic
# ----------------
try:
    from pydantic import BaseModel, Field  # type: ignore

    PYDANTIC_AVAILABLE = True
except Exception:  # pragma: no cover
    PYDANTIC_AVAILABLE = False

    class BaseModel:  # type: ignore
        pass

    def Field(*args, **kwargs):  # type: ignore
        return None


# ---------------
# Logging setup
# ---------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("rationalai.clg")


# ---------------
# Config
# ---------------
FMP_API_KEY = os.environ.get("FMP_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

OPENAI_MODEL_EXTRACTION = os.environ.get("OPENAI_MODEL_EXTRACTION", "gpt-5.1")
OPENAI_MODEL_CANDIDATES = os.environ.get("OPENAI_MODEL_CANDIDATES", "gpt-5.1")
OPENAI_MODEL_VALIDATION = os.environ.get("OPENAI_MODEL_VALIDATION", "gpt-5.1")

logger.info(
    "Using OpenAI models: extraction=%s | candidates=%s | validation=%s",
    OPENAI_MODEL_EXTRACTION,
    OPENAI_MODEL_CANDIDATES,
    OPENAI_MODEL_VALIDATION,
)

if not FMP_API_KEY:
    logger.warning("FMP_API_KEY not set; FMP API calls will fail.")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not set; OpenAI calls will fail.")

# Candidate generation controls
DEFAULT_N_PER_ROLE = 15
EXTRA_LLM_ROUNDS = 2              # additional candidate-generation rounds if too few LLM-approved entities
EXTRA_ROUND_N_PER_ROLE = 5        # candidates per role requested in extra rounds

# Similarity controls
MIN_COMPOSITE_SIM = 0.05

# JSON parsing controls
JSON_REPAIR_ENABLED = True


# ----------------
# HTTP session
# ----------------
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "CLG/2.0"})


# ----------------
# OpenAI client
# ----------------
client = OpenAI(api_key=OPENAI_API_KEY or None)


def backoff_sleep(attempt: int) -> None:
    """Exponential backoff with jitter for retries."""
    base = 0.5
    sleep = base * (2 ** attempt) + random.uniform(0, 0.25)
    time.sleep(min(8.0, sleep))


def normalize_text(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def normalize_keywords(text: str) -> List[str]:
    text = (text or "").lower()
    tokens = re.findall(r"[a-zA-Z0-9+/#-]+", text)
    stop = {
        "and",
        "the",
        "of",
        "for",
        "to",
        "in",
        "on",
        "with",
        "a",
        "an",
        "by",
        "as",
        "from",
        "at",
    }
    return [t for t in tokens if t not in stop and len(t) > 2]


# ----------------
# Data models
# ----------------
class TargetBusinessProfile(BaseModel):
    name: str
    desc: str
    url: Optional[str] = None
    sic: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    ticker: Optional[str] = None

    products_services: List[str] = Field(default_factory=list)
    customer_segments: List[str] = Field(default_factory=list)
    industry_vertical: Optional[str] = None
    business_model: Optional[str] = None
    value_chain_role: Optional[str] = None
    positioning_keywords: List[str] = Field(default_factory=list)

    jobs_to_be_done: List[str] = Field(default_factory=list)
    category_anchors: List[str] = Field(default_factory=list)
    distribution_channels: List[str] = Field(default_factory=list)


class CompetitiveEntity(BaseModel):
    name: Optional[str] = None
    url: Optional[str] = None
    exchange: Optional[str] = None
    ticker: Optional[str] = None

    business_activity: Optional[str] = None
    customer_segment: List[str] = Field(default_factory=list)
    sic_industry: List[str] = Field(default_factory=list)

    raw_sector: Optional[str] = None
    raw_industry: Optional[str] = None

    source_hint: Optional[str] = None
    initial_relationship_hint: Optional[str] = None
    subtheme: Optional[str] = None
    why_fit: Optional[str] = None


# ----------------
# Serialization helpers (Pydantic v1/v2 compatible)
# ----------------
def model_to_dict(obj: Any) -> Dict[str, Any]:
    """Return a plain dict for Pydantic v2, Pydantic v1, or simple objects."""
    if obj is None:
        return {}
    if hasattr(obj, "model_dump"):
        return obj.model_dump()  # type: ignore[attr-defined]
    if hasattr(obj, "dict"):
        return obj.dict()  # type: ignore[attr-defined]
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return {}


def model_to_json(obj: Any, indent: int = 2) -> str:
    if obj is None:
        return "{}"
    if hasattr(obj, "model_dump_json"):
        return obj.model_dump_json(indent=indent)  # type: ignore[attr-defined]
    if hasattr(obj, "json"):
        return obj.json(indent=indent)  # type: ignore[attr-defined]
    return json.dumps(model_to_dict(obj), indent=indent)


# ----------------
# OpenAI helper
# ----------------
def openai_chat(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float = 0.0,
    max_completion_tokens: int = 800,
) -> str:
    """Thin wrapper around Chat Completions."""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set; cannot call OpenAI.")

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:  # pragma: no cover
            logger.warning("OpenAI call failed (attempt %d): %s", attempt + 1, e)
            backoff_sleep(attempt)

    raise RuntimeError("OpenAI chat call failed after retries")


# ----------------
# JSON utilities (robust)
# ----------------
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def extract_json_candidate(text: str) -> str:
    """Try to extract a JSON object/array from a model response."""
    raw = (text or "").strip()
    if not raw:
        return raw

    # If fenced in ```json ...```
    m = _JSON_FENCE_RE.search(raw)
    if m:
        candidate = m.group(1).strip()
        if candidate:
            return candidate

    # Heuristic: find first '{' or '[' and last matching '}' or ']'
    # This is intentionally simple—works well in practice.
    first_curly = raw.find("{")
    first_square = raw.find("[")

    if first_curly == -1 and first_square == -1:
        return raw

    start = min([x for x in [first_curly, first_square] if x != -1])
    tail = raw[start:]

    # try to cut at last brace/bracket
    last_curly = tail.rfind("}")
    last_square = tail.rfind("]")
    end = max(last_curly, last_square)
    if end != -1:
        return tail[: end + 1].strip()

    return tail.strip()


def parse_json_strict(text: str) -> Any:
    return json.loads(text)


def try_parse_json(text: str) -> Tuple[Optional[Any], str]:
    """Attempt strict parse, then extracted parse."""
    raw = (text or "").strip()
    if not raw:
        return None, "empty"

    try:
        return parse_json_strict(raw), "strict"
    except Exception:
        pass

    candidate = extract_json_candidate(raw)
    if candidate != raw:
        try:
            return parse_json_strict(candidate), "extracted"
        except Exception:
            pass

    return None, "failed"


def json_repair_via_llm(raw_text: str, model: str) -> str:
    """Ask the LLM to repair JSON. Returns repaired JSON text or empty list JSON."""
    repair_prompt = (
        "You are a strict JSON repair tool.\n"
        "Return ONLY valid JSON (no backticks, no commentary).\n"
        "If the input is empty, return an empty JSON array: [].\n\n"
        "INPUT:\n" + (raw_text or "")
    )
    messages = [
        {"role": "system", "content": "You output only valid JSON."},
        {"role": "user", "content": repair_prompt},
    ]
    try:
        return openai_chat(messages, model=model, temperature=0.0, max_completion_tokens=900)
    except Exception as e:  # pragma: no cover
        logger.warning("JSON repair call failed: %s", e)
        return "[]"


def load_json_with_repair(raw_text: str, *, model: str) -> Any:
    """Parse JSON with extraction and optional repair."""
    data, mode = try_parse_json(raw_text)
    if data is not None:
        return data

    if not JSON_REPAIR_ENABLED:
        raise json.JSONDecodeError("Invalid JSON", raw_text, 0)

    repaired = json_repair_via_llm(raw_text, model=model)
    data2, mode2 = try_parse_json(repaired)
    if data2 is not None:
        return data2

    raise json.JSONDecodeError("Invalid JSON even after repair", repaired, 0)


# ----------------
# LLM prompts (correct formatting, minimal assumptions)
# ----------------
EXTRACT_TARGET_PROFILE_PROMPT = (
    "You are a senior competitive strategy analyst.\n\n"
    "Extract a decision-ready TARGET PROFILE from the description.\n"
    "Be concise and do NOT invent facts.\n\n"
    "Return STRICT JSON ONLY with this schema:\n"
    "{\n"
    '  "products_services": [string],\n'
    '  "primary_customer_segments": [string],\n'
    '  "jobs_to_be_done": [string],\n'
    '  "business_model": string|null,\n'
    '  "distribution_channels": [string],\n'
    '  "value_chain_role": string|null,\n'
    '  "category_anchors": [string],\n'
    '  "positioning_keywords": [string]\n'
    "}\n\n"
    "Rules:\n"
    "- products_services: 8-15 short phrases.\n"
    "- primary_customer_segments: 5-10.\n"
    "- jobs_to_be_done: 3-6 statements like 'helps <customer> do <job> by <mechanism>'.\n"
    "- category_anchors: 2-4 category labels.\n"
    "- If uncertain, use null or empty list.\n"
    "- JSON only.\n\n"
    "TARGET COMPANY DESCRIPTION:\n"
)

EXTRACT_SEGMENTS_FROM_PROFILE_PROMPT = (
    "You are analyzing a PUBLIC company description.\n"
    "From the text below, extract 3-10 concise CUSTOMER SEGMENT labels that describe\n"
    "the major types of clients or industries the company serves.\n\n"
    "Return STRICT JSON:\n"
    '{ "customer_segments": [ ... ] }\n\n'
    "If you cannot identify customer segments, use an empty list.\n\n"
    "TEXT:\n"
)

LLM_DIRECT_COMPETITOR_PROMPT = (
    "You are a senior competitive intelligence analyst.\n\n"
    "You will be given structured information about a TARGET COMPANY.\n"
    "Your task is to identify companies that are DIRECT COMPETITORS.\n\n"
    "Definition — direct_competitor:\n"
    "A company that offers the SAME core product/service, to the SAME primary customer,\n"
    "for the SAME primary workflow/use case.\n\n"
    "Return ONLY a JSON array of exactly N companies.\n\n"
    "Rules:\n"
    "- Companies must be real operating businesses.\n"
    "- Avoid duplicates/near-duplicates.\n"
    "- If unsure a company exists, do not include it.\n"
    "- If website unknown, set null.\n"
    "- Output JSON only.\n\n"
    "Schema (return exactly this shape):\n"
    "[\n"
    "  {\n"
    "    \"name\": string,\n"
    "    \"website\": string|null,\n"
    "    \"relationship_hint\": \"direct_competitor\",\n"
    "    \"subtheme\": string,\n"
    "    \"why_fit\": string\n"
    "  }\n"
    "]\n\n"
    "N: __N__\n\n"
    "TARGET COMPANY JSON:\n"
)

LLM_ADJACENT_COMPETITOR_PROMPT = (
    "You are a senior competitive intelligence analyst.\n\n"
    "You will be given structured information about a TARGET COMPANY.\n"
    "Your task is to identify ADJACENT COMPETITORS.\n\n"
    "Definition — adjacent_competitor:\n"
    "Overlaps with the target in customer segment OR overlaps in product module/workflow,\n"
    "but not fully the same in both. These are partial substitutes or bundling/expansion threats.\n\n"
    "Return ONLY a JSON array of exactly N companies.\n\n"
    "Rules:\n"
    "- Companies must be real operating businesses.\n"
    "- Avoid duplicates/near-duplicates.\n"
    "- If unsure a company exists, do not include it.\n"
    "- If website unknown, set null.\n"
    "- Output JSON only.\n\n"
    "Schema:\n"
    "[\n"
    "  {\n"
    "    \"name\": string,\n"
    "    \"website\": string|null,\n"
    "    \"relationship_hint\": \"adjacent_competitor\",\n"
    "    \"subtheme\": string,\n"
    "    \"why_fit\": string\n"
    "  }\n"
    "]\n\n"
    "N: __N__\n\n"
    "TARGET COMPANY JSON:\n"
)

LLM_SUBSTITUTE_PROMPT = (
    "You are a senior competitive intelligence analyst.\n\n"
    "You will be given structured information about a TARGET COMPANY.\n"
    "Your task is to identify SUBSTITUTES.\n\n"
    "Definition — substitute:\n"
    "Solves the SAME underlying customer problem/job-to-be-done as the target,\n"
    "but via a DIFFERENT category, product type, or business model.\n\n"
    "Return ONLY a JSON array of exactly N companies.\n\n"
    "Rules:\n"
    "- Companies must be real operating businesses.\n"
    "- Avoid duplicates/near-duplicates.\n"
    "- If unsure a company exists, do not include it.\n"
    "- If website unknown, set null.\n"
    "- Output JSON only.\n\n"
    "Schema:\n"
    "[\n"
    "  {\n"
    "    \"name\": string,\n"
    "    \"website\": string|null,\n"
    "    \"relationship_hint\": \"substitute\",\n"
    "    \"subtheme\": string,\n"
    "    \"why_fit\": string\n"
    "  }\n"
    "]\n\n"
    "N: __N__\n\n"
    "TARGET COMPANY JSON:\n"
)

LLM_SUPPLIER_PROMPT = (
    "You are a senior competitive intelligence analyst.\n\n"
    "You will be given structured information about a TARGET COMPANY.\n"
    "Your task is to identify SUPPLIERS.\n\n"
    "Definition — supplier:\n"
    "Provides critical inputs, infrastructure, or enabling services that the target depends on\n"
    "OR that the target's customers depend on to complete their workflow.\n\n"
    "Return ONLY a JSON array of exactly N companies.\n\n"
    "Rules:\n"
    "- Companies must be real operating businesses.\n"
    "- Avoid duplicates/near-duplicates.\n"
    "- If unsure a company exists, do not include it.\n"
    "- If website unknown, set null.\n"
    "- Output JSON only.\n\n"
    "Schema:\n"
    "[\n"
    "  {\n"
    "    \"name\": string,\n"
    "    \"website\": string|null,\n"
    "    \"relationship_hint\": \"supplier\",\n"
    "    \"subtheme\": string,\n"
    "    \"why_fit\": string\n"
    "  }\n"
    "]\n\n"
    "N: __N__\n\n"
    "TARGET COMPANY JSON:\n"
)

LLM_COMPETITIVE_RELATIONSHIP_PROMPT = (
    "You are a competitive strategy analyst.\n\n"
    "Classify the relationship between TARGET and CANDIDATE.\n"
    "Use ONLY these labels:\n"
    "- direct_competitor\n"
    "- adjacent_competitor\n"
    "- substitute\n"
    "- supplier\n"
    "- downstream_customer\n"
    "- unrelated\n\n"
    "Decision rule:\n"
    "1) If same primary customer + same primary workflow + overlapping core product => direct_competitor\n"
    "2) Else if partial overlap in customers OR overlapping module/workflow => adjacent_competitor\n"
    "3) Else if solves same jobs-to-be-done via different category => substitute\n"
    "4) Else if provides enabling infra/inputs used by target or target customers => supplier\n"
    "5) Else if buys/distributes target => downstream_customer\n"
    "6) Else => unrelated\n\n"
    "Use initial_relationship_hint only as a weak prior.\n"
    "Do not use size, geography, valuation.\n\n"
    "Return STRICT JSON ONLY:\n"
    "{\n"
    '  "category": "<one allowed label>",\n'
    '  "confidence": 0.0-1.0,\n'
    '  "reason": "1-2 sentences focusing on product + customer + workflow"\n'
    "}\n\n"
    "TARGET COMPANY JSON:\n"
    "CANDIDATE COMPANY JSON:\n"
)


# ----------------
# FMP helpers
# ----------------
FMP_BASE = "https://financialmodelingprep.com/stable"


def fmp_get(url_path: str, params: Dict[str, Any]) -> Any:
    params = {**params, "apikey": FMP_API_KEY}
    for attempt in range(3):
        try:
            resp = SESSION.get(f"{FMP_BASE}{url_path}", params=params, timeout=20)
            if resp.status_code == 429 or resp.status_code >= 500:
                logger.warning("FMP rate/server issue (%s): %s", resp.status_code, resp.text[:160])
                backoff_sleep(attempt)
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as e:  # pragma: no cover
            logger.warning("FMP call failed (attempt %d): %s", attempt + 1, e)
            backoff_sleep(attempt)
    return None


def fmp_search_company(query: str) -> List[Dict[str, Any]]:
    if not FMP_API_KEY:
        return []
    data = fmp_get("/search-name", {"query": query, "limit": 20, "exchange": ""}) or []
    return data if isinstance(data, list) else []


def fmp_profile_by_symbol(symbol: str) -> List[Dict[str, Any]]:
    if not FMP_API_KEY:
        return []
    data = fmp_get(f"/profile?symbol={symbol}", {}) or []
    return data if isinstance(data, list) else []


# ----------------
# Target profile extraction
# ----------------
def extract_target_business_fields(desc: str) -> Tuple[List[str], List[str], List[str], List[str], List[str], Optional[str], Optional[str]]:
    if not OPENAI_API_KEY:
        tokens = normalize_keywords(desc)
        return tokens[:20], [], [], [], [], None, None

    messages = [
        {"role": "system", "content": "Return only valid JSON."},
        {"role": "user", "content": EXTRACT_TARGET_PROFILE_PROMPT + (desc or "")},
    ]

    try:
        raw = openai_chat(messages, OPENAI_MODEL_EXTRACTION, temperature=0.0, max_completion_tokens=900)
        data = load_json_with_repair(raw, model=OPENAI_MODEL_EXTRACTION)
        if not isinstance(data, dict):
            raise ValueError("Extraction JSON was not an object")

        return (
            list(data.get("products_services", []) or []),
            list(data.get("primary_customer_segments", []) or []),
            list(data.get("jobs_to_be_done", []) or []),
            list(data.get("category_anchors", []) or []),
            list(data.get("distribution_channels", []) or []),
            data.get("business_model"),
            data.get("value_chain_role"),
        )
    except Exception as e:
        logger.warning("Extraction failed: %s", e)
        tokens = normalize_keywords(desc)
        return tokens[:20], [], [], [], [], None, None


def llm_extract_segments_from_profile(text: str) -> List[str]:
    if not OPENAI_API_KEY or not text:
        return []

    messages = [
        {"role": "system", "content": "You output only valid JSON."},
        {"role": "user", "content": EXTRACT_SEGMENTS_FROM_PROFILE_PROMPT + text},
    ]

    try:
        raw = openai_chat(messages, OPENAI_MODEL_EXTRACTION, temperature=0.0, max_completion_tokens=600)
        data = load_json_with_repair(raw, model=OPENAI_MODEL_EXTRACTION)
        if not isinstance(data, dict):
            return []
        segs = data.get("customer_segments", []) or []
        return [str(x).strip() for x in segs if isinstance(x, str) and x.strip()][:10]
    except Exception as e:  # pragma: no cover
        logger.warning("Candidate segment extraction failed: %s", e)
        return []


def build_target_business_profile(name: str, desc: str, url: Optional[str], sic: Optional[str]) -> TargetBusinessProfile:
    products, segments, jobs, anchors, channels, business_model, value_chain_role = extract_target_business_fields(desc)

    return TargetBusinessProfile(
        name=name,
        desc=desc,
        url=url,
        sic=sic,
        products_services=products,
        customer_segments=segments,
        jobs_to_be_done=jobs,
        category_anchors=anchors,
        distribution_channels=channels,
        business_model=business_model,
        value_chain_role=value_chain_role,
    )


def find_target_in_fmp(name: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    hits = fmp_search_company(name)
    if not hits:
        return None, None, None

    name_lower = name.lower()
    sorted_hits = sorted(hits, key=lambda h: 0 if normalize_text(h.get("name")) == name_lower else 1)

    for h in sorted_hits:
        sym = h.get("symbol")
        if not sym:
            continue
        prof = fmp_profile_by_symbol(sym)
        if prof:
            p = prof[0]
            return sym, p.get("sector"), p.get("industry")

    return None, None, None


# ----------------
# Candidate generation (per role)
# ----------------
ROLE_TO_PROMPT = {
    "direct_competitor": LLM_DIRECT_COMPETITOR_PROMPT,
    "adjacent_competitor": LLM_ADJACENT_COMPETITOR_PROMPT,
    "substitute": LLM_SUBSTITUTE_PROMPT,
    "supplier": LLM_SUPPLIER_PROMPT,
}


def generate_candidates_for_role(target: TargetBusinessProfile, role: str, n: int) -> List[Dict[str, Any]]:
    """Generate candidates for a single role with strict JSON output."""
    if role not in ROLE_TO_PROMPT:
        raise ValueError(f"Unknown role: {role}")

    if not OPENAI_API_KEY:
        return []

    prompt = ROLE_TO_PROMPT[role].replace("__N__", str(int(n)))

    payload = model_to_dict(target)
    # Keep target payload compact and stable
    payload = {
        "name": payload.get("name"),
        "description": payload.get("desc") or payload.get("description"),
        "homepage_url": payload.get("url"),
        "primary_sic_text": payload.get("sic"),
        "sector": payload.get("sector"),
        "industry": payload.get("industry"),
        "products_services": payload.get("products_services", []),
        "customer_segments": payload.get("customer_segments", []),
        "jobs_to_be_done": payload.get("jobs_to_be_done", []),
        "category_anchors": payload.get("category_anchors", []),
        "business_model": payload.get("business_model"),
        "value_chain_role": payload.get("value_chain_role"),
        "distribution_channels": payload.get("distribution_channels", []),
        "positioning_keywords": payload.get("positioning_keywords", []),
    }

    messages = [
        {"role": "system", "content": "You output only valid JSON."},
        {"role": "user", "content": prompt + json.dumps(payload, indent=2)},
    ]

    raw = openai_chat(messages, model=OPENAI_MODEL_CANDIDATES, temperature=0.0, max_completion_tokens=1600)

    data = load_json_with_repair(raw, model=OPENAI_MODEL_CANDIDATES)
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return []

    out: List[Dict[str, Any]] = []
    for obj in data:
        if not isinstance(obj, dict):
            continue
        nm = (obj.get("name") or "").strip()
        if not nm:
            continue
        out.append(
            {
                "name": nm,
                "website": (obj.get("website") or None),
                "relationship_hint": role,
                "subtheme": (obj.get("subtheme") or None),
                "why_fit": (obj.get("why_fit") or None),
            }
        )

    # Deduplicate by normalized name
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for c in out:
        key = normalize_text(c.get("name"))
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(c)

    return deduped[: max(0, int(n))]


def generate_competitive_candidates(target: TargetBusinessProfile, n_per_role: int) -> List[Dict[str, Any]]:
    """Generate candidates across roles, then combine."""
    combined: List[Dict[str, Any]] = []

    for role in ["direct_competitor", "adjacent_competitor", "substitute", "supplier"]:
        try:
            batch = generate_candidates_for_role(target, role=role, n=n_per_role)
            logger.info("LLM returned %d candidates for role=%s", len(batch), role)
            combined.extend(batch)
        except Exception as e:
            logger.warning("Candidate generation failed for role=%s: %s", role, e)

    # Dedup overall by name
    seen = set()
    final: List[Dict[str, Any]] = []
    for c in combined:
        key = normalize_text(c.get("name"))
        if not key or key in seen:
            continue
        seen.add(key)
        final.append(c)

    return final


# ----------------
# Candidate resolution & enrichment
# ----------------
def candidates_from_llm_list(raw_list: List[Dict[str, Any]]) -> List[CompetitiveEntity]:
    out: List[CompetitiveEntity] = []
    for obj in raw_list:
        name = (obj.get("name") or "").strip()
        if not name:
            continue

        hits = fmp_search_company(name)
        if not hits:
            continue

        h0 = hits[0]
        out.append(
            CompetitiveEntity(
                name=name,
                url=obj.get("website"),
                ticker=h0.get("symbol"),
                exchange=h0.get("exchangeShortName"),
                initial_relationship_hint=obj.get("relationship_hint"),
                subtheme=obj.get("subtheme"),
                why_fit=obj.get("why_fit"),
                source_hint=f"llm-candidates:{obj.get('relationship_hint')}",
            )
        )

    return out


def enrich_competitive_entity(c: CompetitiveEntity) -> CompetitiveEntity:
    # Resolve ticker by name if needed
    if not c.ticker and c.name:
        hits = fmp_search_company(c.name)
        if hits:
            sym = hits[0].get("symbol")
            if sym:
                c.ticker = sym

    # Pull full profile
    if c.ticker:
        prof = fmp_profile_by_symbol(c.ticker)
        if prof:
            p = prof[0]
            c.name = p.get("companyName") or c.name
            c.url = p.get("website") or c.url
            c.exchange = p.get("exchangeShortName") or p.get("exchange") or c.exchange
            c.business_activity = p.get("description") or c.business_activity
            c.raw_sector = p.get("sector") or c.raw_sector
            c.raw_industry = p.get("industry") or c.raw_industry
            if c.raw_industry:
                c.sic_industry = [c.raw_industry]

    # Optional: extract customer segments
    if c.business_activity and not c.customer_segment:
        segs = llm_extract_segments_from_profile(c.business_activity)
        if segs:
            c.customer_segment = segs[:10]

    return c


# ----------------
# Scoring
# ----------------
def compute_similarity_tf_idf(target_text: str, candidate_text: str) -> float:
    target_text = (target_text or "").strip()
    candidate_text = (candidate_text or "").strip()
    if not target_text or not candidate_text:
        return 0.0

    try:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform([target_text, candidate_text])
        sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(sim)
    except Exception as e:  # pragma: no cover
        logger.warning("TF-IDF similarity computation failed: %s", e)
        return 0.0


def industry_similarity_score(target: TargetBusinessProfile, c: CompetitiveEntity) -> float:
    t_sector = normalize_text(target.sector)
    t_ind = normalize_text(target.industry)
    c_sector = normalize_text(c.raw_sector)
    c_ind = normalize_text(c.raw_industry)

    if t_ind and c_ind and t_ind == c_ind:
        return 1.0
    if t_sector and c_sector and t_sector == c_sector:
        return 0.7

    tokens_t = set(normalize_keywords(" ".join([t_sector, t_ind]).strip()))
    tokens_c = set(normalize_keywords(" ".join([c_sector, c_ind]).strip()))
    if tokens_t and tokens_c:
        overlap = len(tokens_t & tokens_c) / max(1, len(tokens_t | tokens_c))
        if overlap > 0.5:
            return 0.4
        if overlap > 0.0:
            return 0.25

    return 0.1 if (c_sector or c_ind) else 0.0


def compute_competitive_proximity(target: TargetBusinessProfile, cand: CompetitiveEntity) -> Tuple[float, float, float]:
    tfidf_sim = compute_similarity_tf_idf(target.desc, cand.business_activity or "")
    ind_sim = industry_similarity_score(target, cand)
    composite = 0.7 * tfidf_sim + 0.3 * ind_sim
    return tfidf_sim, ind_sim, composite


def validate_entity_identity(c: CompetitiveEntity) -> bool:
    return bool(c.ticker)


# ----------------
# LLM relationship classification
# ----------------
def classify_competitive_relationship(target: TargetBusinessProfile, candidates: List[CompetitiveEntity]) -> Dict[str, Dict[str, Any]]:
    if not OPENAI_API_KEY or not candidates:
        return {}

    results: Dict[str, Dict[str, Any]] = {}

    target_json = {
        "name": target.name,
        "description": target.desc,
        "homepage_url": target.url,
        "primary_sic_text": target.sic,
        "sector": target.sector,
        "industry": target.industry,
        "customer_segments": target.customer_segments,
        "products_services": target.products_services,
        "jobs_to_be_done": target.jobs_to_be_done,
        "category_anchors": target.category_anchors,
        "business_model": target.business_model,
        "value_chain_role": target.value_chain_role,
        "positioning_keywords": target.positioning_keywords,
    }

    for c in candidates:
        tkr = (c.ticker or "").upper()
        if not tkr:
            continue

        candidate_json = {
            "name": c.name,
            "ticker": c.ticker,
            "exchange": c.exchange,
            "business_activity": c.business_activity,
            "sector": c.raw_sector,
            "industry": c.raw_industry,
            "customer_segments": c.customer_segment,
            "sic_industry": ", ".join(c.sic_industry) if c.sic_industry else None,
            "initial_relationship_hint": c.initial_relationship_hint,
            "subtheme": c.subtheme,
            "why_fit": c.why_fit,
        }

        messages = [
            {"role": "system", "content": "You output only valid JSON."},
            {
                "role": "user",
                "content": (
                    LLM_COMPETITIVE_RELATIONSHIP_PROMPT
                    + json.dumps(target_json, indent=2)
                    + "\n\nCANDIDATE COMPANY JSON:\n"
                    + json.dumps(candidate_json, indent=2)
                ),
            },
        ]

        try:
            raw = openai_chat(messages, model=OPENAI_MODEL_VALIDATION, temperature=0.0, max_completion_tokens=650)
            data = load_json_with_repair(raw, model=OPENAI_MODEL_VALIDATION)
            if not isinstance(data, dict):
                raise ValueError("Classification JSON was not an object")

            category = str(data.get("category", "")).strip() or "unclassified"
            reason = str(data.get("reason", "")).strip()
            confidence = data.get("confidence", None)

            results[tkr] = {
                "category": category,
                "reason": reason,
                "confidence": confidence,
            }

        except Exception as e:  # pragma: no cover
            logger.warning("LLM relationship classification failed for %s: %s", tkr, e)
            results[tkr] = {
                "category": "unclassified",
                "reason": "LLM classification failed; keeping entity based on score only.",
                "confidence": None,
            }

    return results


# ----------------
# Output formatting
# ----------------
def print_grouped_competitive_landscape(df: pd.DataFrame) -> None:
    if df.empty or "competitive_role" not in df.columns:
        print("No competitive roles available to display.")
        return

    role_map = {
        "direct_competitor": "Direct Competitors",
        "adjacent_competitor": "Adjacent Competitors",
        "substitute": "Substitutes",
        "supplier": "Suppliers",
        "downstream_customer": "Downstream Customers",
        "unrelated": "Unrelated",
        "unclassified": "Unclassified",
    }

    print("\n" + "=" * 60)
    print("COMPETITIVE LANDSCAPE (STRATEGIC VIEW)")
    print("=" * 60)

    for role_key, role_label in role_map.items():
        subset = df[df["competitive_role"] == role_key]
        if subset.empty:
            continue

        print(f"\n{role_label}:")
        for _, row in subset.iterrows():
            name = row.get("name")
            ticker = row.get("ticker")
            exchange = row.get("exchange")
            url = row.get("url")

            line = f"  - {name}"
            if ticker:
                line += f" ({ticker}"
                if exchange:
                    line += f", {exchange}"
                line += ")"
            if url:
                line += f" → {url}"
            print(line)


# ----------------
# Main pipeline
# ----------------
def build_competitive_landscape(target: TargetBusinessProfile, outdir: str, max_final: int = 40, n_per_role: int = DEFAULT_N_PER_ROLE) -> pd.DataFrame:
    os.makedirs(outdir, exist_ok=True)

    # Step A: locate target in FMP
    ticker, sector, industry = find_target_in_fmp(target.name)
    target.ticker = ticker
    target.sector, target.industry = sector, industry
    logger.info("Target ticker=%s sector=%s industry=%s", ticker, sector, industry)

    # Step B: generate candidates (per role)
    raw_llm_candidates = generate_competitive_candidates(target, n_per_role=n_per_role)
    logger.info("LLM returned %d total raw candidate entries.", len(raw_llm_candidates))

    initial = candidates_from_llm_list(raw_llm_candidates)
    logger.info("After resolving tickers, %d competitive entities remain.", len(initial))

    if not initial:
        logger.warning("No initial candidates from LLM list. Cannot build competitive landscape.")
        return pd.DataFrame()

    # Step C: deduplicate by (ticker, name)
    dedup: Dict[Tuple[str, str], CompetitiveEntity] = {}
    for c in initial:
        key = ((c.ticker or "").upper(), normalize_text(c.name))
        if key not in dedup:
            dedup[key] = c
    candidates = list(dedup.values())

    seen_tickers = set((c.ticker or "").upper() for c in candidates if c.ticker)

    # Step D: enrich
    enriched: List[CompetitiveEntity] = []
    for c in candidates:
        try:
            enriched.append(enrich_competitive_entity(c))
        except Exception as e:
            logger.warning("Enrichment failed for %s: %s", c.name or c.ticker, e)

    # Step E: validate + score
    rows: List[Dict[str, Any]] = []
    scored_candidates: List[CompetitiveEntity] = []
    target_ticker_norm = (target.ticker or "").upper()

    for c in enriched:
        if (c.ticker or "").upper() == target_ticker_norm:
            continue
        if not validate_entity_identity(c):
            continue

        tfidf_sim, ind_sim, composite = compute_competitive_proximity(target, c)
        if composite < MIN_COMPOSITE_SIM:
            continue

        scored_candidates.append(c)
        rows.append(
            {
                "name": c.name,
                "url": c.url,
                "exchange": c.exchange,
                "ticker": c.ticker,
                "business_activity": c.business_activity,
                "customer_segment": ", ".join(c.customer_segment) if c.customer_segment else None,
                "SIC_industry": ", ".join(c.sic_industry) if c.sic_industry else None,
                "_raw_sector": c.raw_sector,
                "_raw_industry": c.raw_industry,
                "_source": c.source_hint,
                "_initial_relationship_hint": c.initial_relationship_hint,
                "_subtheme": c.subtheme,
                "_why_fit": c.why_fit,
                "_tfidf_similarity": round(tfidf_sim, 4),
                "_industry_similarity": round(ind_sim, 4),
                "_competitive_proximity": round(composite, 4),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("No competitive entities passed validation/scoring.")
        return df

    df = df.sort_values(by=["_competitive_proximity", "_tfidf_similarity", "_industry_similarity"], ascending=False)
    df = df.head(max_final)

    # Step F: relationship classification
    tickers_top = [str(t).upper() for t in df["ticker"].tolist() if t]
    c_map = {(c.ticker or "").upper(): c for c in scored_candidates}
    to_classify = [c_map[t] for t in tickers_top if t in c_map]

    classification_results = classify_competitive_relationship(target, to_classify)

    df["competitive_role"] = [
        classification_results.get(str(t).upper(), {}).get("category", "unclassified")
        for t in df["ticker"].tolist()
    ]
    df["strategic_reason"] = [
        classification_results.get(str(t).upper(), {}).get("reason", None)
        for t in df["ticker"].tolist()
    ]
    df["confidence"] = [
        classification_results.get(str(t).upper(), {}).get("confidence", None)
        for t in df["ticker"].tolist()
    ]

    # Step G: if too few strategically relevant, run extra rounds
    def _is_relevant(cat: str) -> bool:
        return (cat or "").strip() not in {"unrelated", "unclassified", ""}

    approved = df[df["competitive_role"].apply(lambda x: _is_relevant(str(x)))].copy()
    approved = approved.drop_duplicates(subset=["ticker"], keep="first")

    if len(approved) < 3 and EXTRA_LLM_ROUNDS > 0:
        logger.info("Only %d relevant entities; running extra rounds.", len(approved))

        all_approved_frames: List[pd.DataFrame] = [approved]

        for round_idx in range(EXTRA_LLM_ROUNDS):
            logger.info("Starting extra LLM round %d", round_idx + 1)

            raw_extra = generate_competitive_candidates(target, n_per_role=EXTRA_ROUND_N_PER_ROLE)
            initial_extra = candidates_from_llm_list(raw_extra)
            if not initial_extra:
                continue

            # Dedup vs seen tickers
            kept: List[CompetitiveEntity] = []
            for c in initial_extra:
                tkr = (c.ticker or "").upper()
                if not tkr or tkr in seen_tickers:
                    continue
                seen_tickers.add(tkr)
                kept.append(c)

            if not kept:
                continue

            # Enrich
            enriched_extra: List[CompetitiveEntity] = []
            for c in kept:
                try:
                    enriched_extra.append(enrich_competitive_entity(c))
                except Exception as e:
                    logger.warning("Enrichment failed (extra) for %s: %s", c.name or c.ticker, e)

            # Score
            rows_extra: List[Dict[str, Any]] = []
            scored_extra: List[CompetitiveEntity] = []
            for c in enriched_extra:
                if not validate_entity_identity(c):
                    continue
                if (c.ticker or "").upper() == target_ticker_norm:
                    continue

                tfidf_sim, ind_sim, composite = compute_competitive_proximity(target, c)
                if composite < MIN_COMPOSITE_SIM:
                    continue

                scored_extra.append(c)
                rows_extra.append(
                    {
                        "name": c.name,
                        "url": c.url,
                        "exchange": c.exchange,
                        "ticker": c.ticker,
                        "business_activity": c.business_activity,
                        "customer_segment": ", ".join(c.customer_segment) if c.customer_segment else None,
                        "SIC_industry": ", ".join(c.sic_industry) if c.sic_industry else None,
                        "_raw_sector": c.raw_sector,
                        "_raw_industry": c.raw_industry,
                        "_source": c.source_hint,
                        "_initial_relationship_hint": c.initial_relationship_hint,
                        "_subtheme": c.subtheme,
                        "_why_fit": c.why_fit,
                        "_tfidf_similarity": round(tfidf_sim, 4),
                        "_industry_similarity": round(ind_sim, 4),
                        "_competitive_proximity": round(composite, 4),
                    }
                )

            if not rows_extra:
                continue

            df_extra = pd.DataFrame(rows_extra)
            df_extra = df_extra.sort_values(by=["_competitive_proximity", "_tfidf_similarity", "_industry_similarity"], ascending=False)
            df_extra = df_extra.head(max_final)

            tickers_top_extra = [str(t).upper() for t in df_extra["ticker"].tolist() if t]
            c_map_extra = {(c.ticker or "").upper(): c for c in scored_extra}
            to_classify_extra = [c_map_extra[t] for t in tickers_top_extra if t in c_map_extra]

            classification_extra = classify_competitive_relationship(target, to_classify_extra)

            df_extra["competitive_role"] = [
                classification_extra.get(str(t).upper(), {}).get("category", "unclassified")
                for t in df_extra["ticker"].tolist()
            ]
            df_extra["strategic_reason"] = [
                classification_extra.get(str(t).upper(), {}).get("reason", None)
                for t in df_extra["ticker"].tolist()
            ]
            df_extra["confidence"] = [
                classification_extra.get(str(t).upper(), {}).get("confidence", None)
                for t in df_extra["ticker"].tolist()
            ]

            df_extra_approved = df_extra[df_extra["competitive_role"].apply(lambda x: _is_relevant(str(x)))].copy()
            df_extra_approved = df_extra_approved.drop_duplicates(subset=["ticker"], keep="first")

            if not df_extra_approved.empty:
                all_approved_frames.append(df_extra_approved)

            merged = pd.concat(all_approved_frames, ignore_index=True)
            merged = merged.sort_values(by=["_competitive_proximity", "_tfidf_similarity", "_industry_similarity"], ascending=False)
            merged = merged.drop_duplicates(subset=["ticker"], keep="first")

            logger.info("After extra round %d, relevant distinct=%d", round_idx + 1, len(merged))

            if len(merged) >= 3:
                approved = merged
                break

        if len(approved) >= 3:
            df_final = approved.head(max_final)
        else:
            df_final = df
    else:
        df_final = approved.head(max_final) if not approved.empty else df

    # Step H: export deliverable
    required_cols = [
        "name",
        "url",
        "exchange",
        "ticker",
        "business_activity",
        "customer_segment",
        "SIC_industry",
        "competitive_role",
        "strategic_reason",
        "confidence",
        "_competitive_proximity",
        "_tfidf_similarity",
        "_industry_similarity",
        "_subtheme",
        "_why_fit",
    ]

    for col in required_cols:
        if col not in df_final.columns:
            df_final[col] = None

    df_deliverable = df_final[required_cols].copy()

    csv_path = os.path.join(outdir, "competitive_landscape.csv")
    parquet_path = os.path.join(outdir, "competitive_landscape.parquet")
    target_path = os.path.join(outdir, "target_business_profile.json")

    df_deliverable.to_csv(csv_path, index=False)
    try:
        df_deliverable.to_parquet(parquet_path, index=False)
    except Exception as e:  # pragma: no cover
        logger.warning("Parquet export failed (install pyarrow): %s", e)

    try:
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(model_to_json(target, indent=2))
        logger.info("Wrote %s", target_path)
    except Exception as e:
        logger.warning("Failed to write target_business_profile.json: %s", e)

    logger.info("Wrote %s and %s", csv_path, parquet_path)
    return df_deliverable


# ----------------
# CLI
# ----------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Competitive Landscape Generator (LLM + TF-IDF + public market data)")
    p.add_argument("--name", required=True, help="Target company name")
    p.add_argument("--url", default=None, help="Target company homepage URL")
    p.add_argument("--desc", required=True, help="Brief business description")
    p.add_argument("--sic", default=None, help="Primary SIC industry classification (text)")
    p.add_argument("--outdir", default="./runs/output", help="Output directory")
    p.add_argument("--max_final", type=int, default=40)
    p.add_argument("--n_per_role", type=int, default=DEFAULT_N_PER_ROLE, help="Number of candidates to request per role")
    p.add_argument("--no_json_repair", action="store_true", help="Disable LLM JSON repair")
    return p.parse_args()


def main() -> None:
    global JSON_REPAIR_ENABLED

    args = parse_args()
    JSON_REPAIR_ENABLED = not args.no_json_repair

    # Clamp max_final
    max_final = max(4, min(int(args.max_final), 40))
    n_per_role = max(2, min(int(args.n_per_role), 15))

    target = build_target_business_profile(args.name, args.desc, args.url, args.sic)
    logger.info("Target products/services (from extraction): %s", target.products_services)

    df = build_competitive_landscape(target, args.outdir, max_final=max_final, n_per_role=n_per_role)

    if not df.empty:
        print_grouped_competitive_landscape(df)
    else:
        print("No valid competitive entities found.")


if __name__ == "__main__":
    main()
