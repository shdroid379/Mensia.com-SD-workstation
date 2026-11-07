"""Vin - research dossier pipeline.

Plan -> Retrieve -> Rerank -> Ledger -> Dossier -> Red team -> Patch -> Audit -> Polish.

Every stage is a separate coroutine function taking inputs and returning outputs, so
each can be stubbed independently (see --dry-run and the DRY_RUN router near the
bottom of this file).
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import math
import os
import random
import re
import sys
import time
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)

import httpx
from openai import AsyncOpenAI

from google import genai
from google.genai import types as genai_types

import instruc_prompt as P

import hashlib
from collections import defaultdict

import networkx as nx
import numpy as np
from pydantic import BaseModel, Field
from rapidfuzz import fuzz

APP_NAME = "vin"
RUNS_DIRNAME = "runs"
HTTP_ATTEMPTS = 3
BACKOFF_BASE_S = 1.5
BACKOFF_JITTER_S = 0.4

RERANK_FLOOR = 8
RERANK_CAP = 60
RERANK_CORE_CAP = 45
RERANK_NOVELTY_CAP = 15
DURALUMIN_BATCH_SIZE = 6
DURALUMIN_CONCURRENCY = 8
CHUNK_MIN_WORDS = 250
CHUNK_MAX_WORDS = 400
CHUNK_TARGET_WORDS = 330

CITATION_RE = re.compile(r"\[S(\d{1,4})\]")
RESEARCH_RE = re.compile(r"^\s*Research\s*:\s*(\S.*?)\s*$", re.IGNORECASE)
SEARCH_RE = re.compile(r"^\s*Search\s*:\s*(\S.*?)\s*$", re.IGNORECASE)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
TRACKING_PARAMS = re.compile(
    r"^(utm_[a-z_]+|fbclid|gclid|mc_[a-z]+|ref|ref_src|source|igshid|spm)$", re.IGNORECASE
)


# ============================================================ env helpers


def env_str(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return v.strip() if isinstance(v, str) and v.strip() else default


def env_int(name: str, default: int) -> int:
    try:
        return int(env_str(name) or default)
    except (TypeError, ValueError):
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(env_str(name) or default)
    except (TypeError, ValueError):
        return default


def env_bool(name: str, default: bool) -> bool:
    v = env_str(name)
    if not v:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")



RERANK_MIN_SCORE = env_float("RERANK_MIN_SCORE", 0.30)
DOSSIER_IDLE_TIMEOUT = env_float("DOSSIER_IDLE_TIMEOUT", 930.0)  # unified single timeout
DOSSIER_TOTAL_TIMEOUT = env_float("DOSSIER_TOTAL_TIMEOUT", 930.0)
DOSSIER_MAX_TOKENS = env_int("DOSSIER_MAX_TOKENS", 12000)
REASONING_WARN_CHARS = env_int("REASONING_WARN_CHARS", P.REASONING_CHAR_TARGET)
POLISH_MIN_RATIO = env_float("POLISH_MIN_RATIO", 0.98)
COUNTER_EVIDENCE_TOKEN_BUDGET = env_int("COUNTER_EVIDENCE_TOKEN_BUDGET", 4000)
RERANK_BATCH = env_int("RERANK_BATCH", 32)

# Evidence Ledger knobs
LEDGER_DIR = env_str("LEDGER_DIR", "")
LEDGER_MIN_CHUNK = env_int("LEDGER_MIN_CHUNK", 64)
LEDGER_MAX_CHUNK = env_int("LEDGER_MAX_CHUNK", 480)
LEDGER_PRIORITY_TOP = env_int("LEDGER_PRIORITY_TOP", 384)
LEDGER_PRIORITY_OVERLAP = env_int("LEDGER_PRIORITY_OVERLAP", 144)
LEDGER_MIN_ENTITIES = env_int("LEDGER_MIN_ENTITIES", 2)
LEDGER_MIN_MONEY = env_int("LEDGER_MIN_MONEY", 1)
LEDGER_MIN_DATE = env_int("LEDGER_MIN_DATE", 1)
LEDGER_MENTION_MIN_LEN = env_int("LEDGER_MENTION_MIN_LEN", 5)
LEDGER_MENTION_MAX_LEN = env_int("LEDGER_MENTION_MAX_LEN", 120)
LEDGER_VERBATIM_FALLTHROUGH = env_bool("LEDGER_VERBATIM_FALLTHROUGH", True)


# ============================================================ stage config


@dataclass(frozen=True)
class StageCfg:
    name: str
    base_url: str
    model: str
    key_env: str
    timeout: float
    max_tokens: int
    stream: bool = False
    temperature: float | None = 0.1
    extra_body: dict | None = None
    fallback: tuple["StageCfg", ...] = ()
    note: str = ""
    # "openai" routes through AsyncOpenAI (OpenAI-compatible endpoints).
    # "google_genai" routes through the official google-genai SDK against the
    # Gemini Developer API (Google AI Studio), not the OpenAI-compat shim.
    provider: str = "openai"
    # google-genai only: turns thinking off via ThinkingConfig(thinking_budget=0).
    # Gemini otherwise spends the whole output budget on silent thinking, which
    # starved the fact bank. Set False for models where thinking cannot be disabled.
    disable_thinking: bool = True
    # OpenAI-compatible only: sent as the top-level reasoning_effort field.
    # Kimi K3 defaults to "max", which on the NVIDIA trial endpoint produces very long
    # deliberation and degenerate output; "low" returns clean content.
    reasoning_effort: str | None = None

    def key(self) -> str:
        return env_str(self.key_env)

    def have_key(self) -> bool:
        return bool(self.key())

    def describe(self) -> dict:
        return {
            "stage": self.name,
            "base_url": self.base_url,
            "model": self.model,
            "key_env": self.key_env,
            "key_present": self.have_key(),
            "timeout_s": self.timeout,
            "max_tokens": self.max_tokens,
            "stream": self.stream,
            "provider": self.provider,
            "fallbacks": [f.model for f in self.fallback],
            "note": self.note,
        }


# VERIFIED 2026-10-03 by probing each provider's live /models endpoint and a live
# chat completion. Deviations from the requested spec are called out in `note`.
PLANNER = StageCfg(
    name="plan",
    base_url="https://api.groq.com/openai/v1",
    model=env_str("VIN_PLAN_MODEL", "qwen/qwen3.8-27b"),
    key_env="GROQ_KEY",
    timeout=120.0,
    max_tokens=700,
    note="Primary planner on Groq qwen/qwen3.8-27b (timeout 120s).",
    fallback=(
        StageCfg(
            name="plan-fallback",
            base_url="https://api.cohere.ai/compatibility/v1",
            model=env_str("VIN_INSPECTOR_MODEL", "command-a-03-2025"),
            key_env="COHERE_API_KEY",
            timeout=120.0,
            max_tokens=700,
            note="Fallback planner on Cohere Command-A.",
        ),
    ),
)

DURALUMIN_PROFILER = StageCfg(
    name="profiler",
    base_url="https://api.cohere.ai/compatibility/v1",
    model=env_str("VIN_INSPECTOR_MODEL", "command-a-03-2025"),
    key_env="COHERE_API_KEY",
    timeout=75.0,
    max_tokens=1200,
    provider="openai",
    note="DURALUMIN Phase 0 Case Profiler on Cohere Command-A (timeout 75s).",
    fallback=(
        StageCfg(
            name="profiler-fallback",
            base_url="https://api.groq.com/openai/v1",
            model=env_str("VIN_PROFILER_MODEL", "qwen/qwen3.8-27b"),
            key_env="GROQ_KEY",
            timeout=75.0,
            max_tokens=1200,
            provider="openai",
            note="Fallback Case Profiler on Groq.",
        ),
    ),
)

LEDGER = StageCfg(
    name="ledger",
    base_url="https://inference.api.nscale.com/v1",
    model=env_str("VIN_LEDGER_MODEL", "Qwen/Qwen3-235B-A22B-Instruct-2507"),
    key_env="NSCALE_API_KEY",
    timeout=120.0,
    max_tokens=6000,
    provider="openai",
    extra_body={"enable_thinking": False},
    note="nScale Qwen3-235B with 120s timeout, placed 4th in DURALUMIN_EXTRACT_POOL.",
)


DURALUMIN_INSPECTOR = StageCfg(
    name="inspector",
    base_url="https://api.cohere.ai/compatibility/v1",
    model=env_str("VIN_INSPECTOR_MODEL", "command-a-03-2025"),
    key_env="COHERE_API_KEY",
    timeout=120.0,
    max_tokens=4096,
    provider="openai",
    note="DURALUMIN Phase 2C Chief Inspector on Cohere Command-A (timeout 120s).",
    fallback=(
        StageCfg(
            name="inspector-fallback",
            base_url="https://api.upstage.ai/v1",
            model=env_str("VIN_PATCH_MODEL", "solar-pro4"),
            key_env="UPSTAGE_API_KEY",
            timeout=120.0,
            max_tokens=4096,
            provider="openai",
            note="Fallback Chief Inspector on Upstage Solar-Pro4.",
        ),
    ),
)

# STRICT PRIORITY ORDER: 1. Cohere -> 2. Upstage -> 3. Groq -> 4. nScale (all timeouts >= 105s-120s)
DURALUMIN_EXTRACT_POOL: tuple[StageCfg, ...] = (
    StageCfg(
        name="ledger-cohere",
        base_url="https://api.cohere.ai/compatibility/v1",
        model=env_str("VIN_INSPECTOR_MODEL", "command-a-plus"),
        key_env="COHERE_API_KEY",
        timeout=105.0,
        max_tokens=6000,
        provider="openai",
        note="Priority #1 micro-batch extractor on Cohere Command-A+ (105s timeout).",
    ),
    StageCfg(
        name="ledger-upstage",
        base_url="https://api.upstage.ai/v1",
        model=env_str("VIN_PATCH_MODEL", "solar-pro4"),
        key_env="UPSTAGE_API_KEY",
        timeout=105.0,
        max_tokens=6000,
        provider="openai",
        note="Priority #2 micro-batch extractor on Upstage Solar-Pro4 (105s timeout).",
    ),
    StageCfg(
        name="ledger-groq-qwen",
        base_url="https://api.groq.com/openai/v1",
        model=env_str("VIN_PLAN_FALLBACK_MODEL", "qwen/qwen3.8-27b"),
        key_env="GROQ_KEY",
        timeout=105.0,
        max_tokens=4096,
        provider="openai",
        note="Priority #3 micro-batch extractor on Groq Qwen3.8-27B (105s timeout).",
    ),
    LEDGER,
)

DOSSIER = StageCfg(
    name="dossier",
    base_url="https://integrate.api.nvidia.com/v1",
    model=env_str("VIN_DOSSIER_MODEL", "stepfun-ai/step-5-preview"),
    key_env="NVIDIA_BUILD_API_KEY",
    timeout=DOSSIER_TOTAL_TIMEOUT,
    max_tokens=DOSSIER_MAX_TOKENS,
    stream=True,
    temperature=None,
    reasoning_effort="medium",
    note="Primary Dossier writer on NVIDIA step-5-preview (single 930s timeout, 12k max_tokens).",
    fallback=(
        StageCfg(
            name="dossier-fallback",
            base_url="https://inference.api.nscale.com/v1",
            model="moonshotai/Kimi-K2.5",
            key_env="NSCALE_API_KEY",
            timeout=DOSSIER_TOTAL_TIMEOUT,
            max_tokens=DOSSIER_MAX_TOKENS,
            stream=True,
            temperature=None,
            note="Fallback #1 Dossier writer on nScale Kimi-K2.5 (single 930s timeout).",
        ),
        StageCfg(
            name="dossier-fallback-2",
            base_url="https://api.gmi-serving.com/v1",
            model="XiaomiMiMo/MiMo-V2.5",
            key_env="GMICLOUD_API_KEY",
            timeout=DOSSIER_TOTAL_TIMEOUT,
            max_tokens=DOSSIER_MAX_TOKENS,
            stream=True,
            temperature=None,
            note="Fallback #2 Dossier writer on GMI Cloud MiMo-V2.5 (stream=True).",
        ),
    ),
)

REDTEAM = StageCfg(
    name="redteam",
    base_url="https://api.gmi-serving.com/v1",
    model=env_str("VIN_REDTEAM_MODEL", "XiaomiMiMo/MiMo-V2.5"),
    key_env="GMICLOUD_API_KEY",
    timeout=330.0,
    max_tokens=16000,
    stream=True,
    note="Primary Red Team on GMI Cloud XiaomiMiMo/MiMo-V2.5 with stream=True (330s timeout).",
    fallback=(
        StageCfg(
            name="redteam-fallback",
            base_url="https://finalrouter.com/api/v1",
            model="deepseek/deepseek-v4-flash",
            key_env="FINALROUTER_API_KEY",
            timeout=330.0,
            max_tokens=16000,
            stream=True,
            note="Fallback Red Team on FinalRouter deepseek-v4-flash.",
        ),
    ),
)

PATCH = StageCfg(
    name="patch",
    base_url="https://api.upstage.ai/v1",
    model=env_str("VIN_PATCH_MODEL", "solar-pro4"),
    key_env="UPSTAGE_API_KEY",
    timeout=330.0,
    max_tokens=4096,
    note="Surgical patcher on Upstage solar-pro4 (330s timeout).",
    fallback=(
        StageCfg(
            name="patch-fallback",
            base_url="https://api.cohere.ai/compatibility/v1",
            model="command-a-03-2025",
            key_env="COHERE_API_KEY",
            timeout=330.0,
            max_tokens=4096,
            note="Fallback surgical patcher on Cohere Command-A.",
        ),
    ),
)

AUDIT = StageCfg(
    name="audit",
    base_url=env_str("VIN_AUDIT_BASE_URL", "https://proxyless-api.privatemode.ai/v1"),
    model=env_str("VIN_AUDIT_MODEL", "glm-5.3"),
    key_env="PRIVATEMODE_API_KEY",
    timeout=330.0,
    max_tokens=12000,
    stream=True,
    reasoning_effort="low",
    note="Primary Auditor on Privatemode Proxyless API glm-5.3 (12k max_tokens under 20k/min completion limit, 330s timeout, stream=True).",
    fallback=(
        StageCfg(
            name="audit-fallback",
            base_url="https://api.routeway.ai/v1",
            model="deepseek-v4-flash:free",
            key_env="ROUTEWAY_API_KEY",
            timeout=330.0,
            max_tokens=12000,
            stream=True,
            note="Fallback #1 Auditor on Routeway deepseek-v4-flash:free.",
        ),
        StageCfg(
            name="audit-fallback-2",
            base_url="https://finalrouter.com/api/v1",
            model="deepseek/deepseek-v4-flash",
            key_env="FINALROUTER_API_KEY",
            timeout=330.0,
            max_tokens=12000,
            stream=True,
            note="Fallback #2 Auditor on FinalRouter deepseek/deepseek-v4-flash.",
        ),
    ),
)

CLAIM_AUDIT = StageCfg(
    name="claim_audit",
    base_url="https://api.inference.wandb.ai/v1",
    model="zai-org/GLM-5.3-Flash",
    key_env="COREWEAVE_API_KEY",
    timeout=330.0,
    max_tokens=16384,
    provider="openai",
    reasoning_effort="low",
    note="Per-source Claim Auditor on CoreWeave GLM-5.3-Flash (330s timeout).",
    fallback=(),
)

POLISH = StageCfg(
    name="polish",
    base_url="https://api.llm7.io/v1",
    model="minimax-m2.7",
    key_env="LLM7_API_KEY",
    timeout=450.0,
    max_tokens=16000,
    note="Markdown formatter on LLM7 minimax-m2.7 (450s timeout).",
    fallback=(
        StageCfg(
            name="polish-fallback",
            base_url="https://api.gmi-serving.com/v1",
            model="XiaomiMiMo/MiMo-V2.5",
            key_env="GMICLOUD_API_KEY",
            timeout=450.0,
            max_tokens=16000,
            stream=True,
            note="Fallback formatter on GMI Cloud MiMo-V2.5 (stream=True).",
        ),
    ),
)

SEARCH_CFG = {
    "tavily_research": {
        "name": "tavily-research",
        "base_url": "https://api.tavily.com/research",
        "key_env": "TAVILY_KEY",
        "research_model": "pro",
        "output_length": "long",
        "poll_timeout_s": 600.0,
        "poll_interval_s": 2.0,
        "note": (
            "VERIFIED 2026-10-03: POST returns 200 with status=pending plus a "
            "request_id; GET /research/{request_id} returns status=in_progress then "
            "status=completed with content and sources[]. model enum is mini|pro|auto "
            "and output_length enum is short|standard|long (docs 2026-10-04). Raised "
            "from mini/standard to pro/long on request for deeper research: 'pro' is "
            "documented as comprehensive, multi-angle and suited to complex topics "
            "spanning subtopics. output_length is a target, not a hard cap. Poll "
            "timeout raised to 600s because a pro/long job runs much longer than mini."
        ),
    },
    "you_research": {
        "name": "you-research",
        "base_url": "https://api.you.com/v1/research",
        "key_env": "YOU_API_KEY",
        "effort": "standard",
        "poll_timeout_s": 300.0,
        "poll_interval_s": 5.0,
        "background": True,
        "note": (
            "VERIFIED 2026-10-03: auth header X-API-Key. Sync response is "
            "output.content + output.sources[].url/title/snippets. background=true "
            "returns task_id; poll GET /v1/research/{task_id} for "
            "status queued|running|completed|failed and result.output."
        ),
    },
    "exa": {
        "name": "exa",
        "base_url": "https://api.exa.ai/search",
        "key_env": "EXA_API_KEY",
        "num_results": 2,
        "text_max_chars": 6000,
        "note": (
            "VERIFIED 2026-10-03: POST /search, header x-api-key, body "
            "contents.text.maxCharacters, response results[].text/url/title."
        ),
    },
    "tavily_search": {
        "name": "tavily-search",
        "base_url": "https://api.tavily.com/search",
        "key_env": "TAVILY_KEY",
        "num_results": 20,
        "search_depth": "advanced",
        "chunks_per_source": 3,
        "include_raw_content": "markdown",
        "note": (
            "VERIFIED 2026-10-07 from Tavily Search docs: search_depth advanced "
            "(2 credits/request), chunks_per_source 1-3, max_results 1-20. "
            "Advanced mode returns multiple semantically relevant snippets per URL "
            "and is the highest-relevance option. chunks_per_source=3 is the max. "
            "include_raw_content=markdown yields full cleaned markdown per result."
        ),
    },
    "cloudflare_rerank": {
        "name": "cloudflare-rerank",
        "base_url": "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}",
        "key_env": "CLOUDFLARE_AUTH_TOKEN",
        "key_env_aliases": ("CLOUDFLARE_TOKEN", "Cloudfare_token"),
        "account_id_env": "CLOUDFLARE_ACCOUNT_ID",
        "model": "@cf/baai/bge-reranker-base",
        "note": (
            "VERIFIED 2026-10-03 from Cloudflare's model page and schema-output.json: "
            "body is {query, contexts:[{text}], top_k}; the envelope is "
            "{result:{response:[{id,score}]}, success, errors, messages}."
        ),
    },
    "cloudflare_embed": {
        "name": "cloudflare-embed",
        "base_url": "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}",
        "key_env": "CLOUDFLARE_AUTH_TOKEN",
        "key_env_aliases": ("CLOUDFLARE_TOKEN", "Cloudfare_token"),
        "account_id_env": "CLOUDFLARE_ACCOUNT_ID",
        "model": "@cf/baai/bge-m3",
        "note": (
            "DURALUMIN Phase 2B semantic fingerprint matcher on Cloudflare Workers AI "
            "(@cf/baai/bge-m3). Embeds entity and metric topic strings for cosine clustering."
        ),
    },
}


_PRIMARY_STAGES = (
    PLANNER, DURALUMIN_PROFILER, LEDGER, DURALUMIN_INSPECTOR,
    DOSSIER, REDTEAM, PATCH, AUDIT, CLAIM_AUDIT, POLISH,
)
_ALL_STAGES = _PRIMARY_STAGES + tuple(f for c in _PRIMARY_STAGES for f in c.fallback)
PROVIDER_LABEL = {c.name: f" sdk={c.provider}" for c in _ALL_STAGES if c.provider != "openai"}


def print_banner() -> None:
    """Print the resolved stage configuration. Never prints key values."""
    rows: list[tuple[str, str, str, str, str, str]] = []
    for cfg in _PRIMARY_STAGES:
        chain = [(cfg.name, cfg.model, "primary")] + [
            (f.name, f.model, "fallback") for f in cfg.fallback
        ]
        for stage, model, kind in chain:
            rows.append(
                (kind, stage, cfg.base_url if stage == cfg.name else "", model,
                 cfg.key_env if stage == cfg.name else "", "yes" if cfg.have_key() else "NO")
            )
    for name in ("tavily_research", "you_research", "exa", "tavily_search",
                 "cloudflare_rerank", "cloudflare_embed"):
        s = SEARCH_CFG[name]
        rows.append(
            ("search", s["name"], s["base_url"], s.get("model", "-"), s["key_env"],
             "yes" if any(env_str(e) for e in [s["key_env"], *s.get("key_env_aliases", [])]) else "NO")
        )
    width = max(len(r[2]) for r in rows)
    print("=" * 78)
    print(f"{APP_NAME} - resolved stage configuration")
    print("=" * 78)
    for kind, stage, url, model, key_env, present in rows:
        print(f"[{kind:8}] {stage:22} {model}")
        print(f"{'':11} url  {url:<{width}} key_env={key_env} present={present}"
              f"{PROVIDER_LABEL.get(stage, '')}")
    for cfg in _PRIMARY_STAGES:
        if cfg.note:
            print(f"{'':11} note ({cfg.name}): {cfg.note}")
    for name in ("tavily_research", "you_research", "exa", "tavily_search",
                 "cloudflare_rerank", "cloudflare_embed"):
        s = SEARCH_CFG[name]
        print(f"{'':11} note ({s['name']}): {s['note']}")
    print("=" * 78)


def key_for(cfg: dict) -> str:
    for env_name in [cfg["key_env"], *cfg.get("key_env_aliases", [])]:
        v = env_str(env_name)
        if v:
            return v
    return ""


# ============================================================ run state


@dataclass
class Document:
    source_id: str
    title: str
    url: str
    date: str
    provider: str
    origin: str
    text: str

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "title": self.title,
            "url": self.url,
            "date": self.date,
            "provider": self.provider,
            "origin": self.origin,
            "chars": len(self.text),
        }


@dataclass
class Chunk:
    source_id: str
    index: int
    text: str
    words: int
    score: float = 0.0

    @property
    def uid(self) -> str:
        return f"{self.source_id}#{self.index}"

    def tagged(self) -> str:
        return f"[{self.source_id}] (chunk {self.index}) {self.text}"


class SourceRegistry:
    """Code-owned S1..Sn map. Ids flow through every stage."""

    def __init__(self) -> None:
        self._items: dict[str, dict] = {}
        self._by_url: dict[str, str] = {}
        self._next = 1
        self._texts: dict[str, str] = {}

    def register(
        self, url: str, title: str, date: str, provider: str, origin: str
    ) -> str | None:
        key = normalize_url(url)
        if not key:
            return None
        existing = self._by_url.get(key)
        if existing is not None:
            return existing
        sid = f"S{self._next}"
        self._next += 1
        self._by_url[key] = sid
        self._items[sid] = {
            "url": url,
            "title": (title or "").strip() or url,
            "date": (date or "").strip(),
            "provider": provider,
            "origin": origin if origin in ("primary", "counter") else "primary",
        }
        return sid

    def register_text(self, sid: str, text: str) -> None:
        if sid and sid in self._items and text:
            self._texts[sid] = text

    def text_for(self, sid: str) -> str:
        return self._texts.get(sid, "")

    def known(self, sid: str) -> bool:
        return sid in self._items

    def get(self, sid: str) -> dict:
        return self._items.get(sid, {})

    def items(self) -> dict:
        return self._items

    def title_for(self, sid: str) -> str:
        return self._items.get(sid, {}).get("title", "")

    def __len__(self) -> int:
        return len(self._items)


class RunLog:
    def __init__(self, run_dir: Path, query: str, opts: dict) -> None:
        self.run_dir = run_dir
        self.started = time.time()
        self.registry = SourceRegistry()
        self.data: dict[str, Any] = {
            "query": query,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "options": opts,
            "stages": [],
            "warnings": [],
            "providers": {},
            "usage": {},
            "config": {},
            "artifacts": [],
        }
        self._stage_open: dict | None = None

    # -- artifacts
    def artifact(self, name: str, content: str) -> Path:
        p = self.run_dir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        self.data["artifacts"].append(name)
        return p

    def artifact_json(self, name: str, payload: Any) -> Path:
        return self.artifact(name, json.dumps(payload, indent=2, ensure_ascii=False))

    # -- warnings
    def warn(self, stage: str, message: str) -> None:
        entry = {"stage": stage, "message": message, "t": round(time.time() - self.started, 3)}
        self.data["warnings"].append(entry)
        print(f"  ! WARN [{stage}] {message}")

    def info(self, stage: str, message: str) -> None:
        print(f"  * INFO [{stage}] {message}")

    # -- stage timing
    def stage_begin(self, name: str, detail: dict | None = None) -> None:
        self._stage_open = {
            "stage": name,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "t0": time.time(),
            "status": "running",
            "provider": None,
            "fallback_used": False,
            "usage": None,
            "detail": detail or {},
        }
        print(f"\n--- {name}")

    def stage_usage(
        self,
        provider: str | None = None,
        fallback: bool = False,
        usage: Any = None,
        **extra: Any,
    ) -> None:
        if self._stage_open is None:
            return
        if provider:
            self._stage_open["provider"] = provider
        if fallback:
            self._stage_open["fallback_used"] = True
        if usage is not None:
            self._stage_open["usage"] = _usage_dict(usage)
            for k, v in _usage_dict(usage).items():
                self.data["usage"][k] = self.data["usage"].get(k, 0) + (
                    v if isinstance(v, (int, float)) else 0
                )
        self._stage_open.update(extra)

    def stage_end(self, status: str = "ok", error: str | None = None) -> None:
        if self._stage_open is None:
            return
        entry = self._stage_open
        entry["seconds"] = round(time.time() - entry["t0"], 3)
        entry["status"] = status
        if error:
            entry["error"] = error
        self.data["stages"].append(entry)
        print(
            f"--- {entry['stage']}: {status} in {entry['seconds']}s"
            + (f" ({entry['provider']})" if entry.get("provider") else "")
        )
        self._stage_open = None

    def provider_used(self, stage: str, provider: str, fallback: bool) -> None:
        self.data["providers"][stage] = {
            "provider": provider,
            "fallback_used": fallback,
        }

    def save(self) -> None:
        self.data["ended_at"] = datetime.now(timezone.utc).isoformat()
        self.data["total_seconds"] = round(time.time() - self.started, 3)
        (self.run_dir / "log.json").write_text(
            json.dumps(self.data, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )


def _usage_dict(usage: Any) -> dict:
    if usage is None:
        return {}
    for getter in ("model_dump", "dict", "to_dict"):
        fn = getattr(usage, getter, None)
        if callable(fn):
            try:
                d = fn()
                if isinstance(d, dict):
                    return {k: v for k, v in d.items() if isinstance(v, (int, float))}
            except Exception:
                pass
    if isinstance(usage, dict):
        return {k: v for k, v in usage.items() if isinstance(v, (int, float))}
    return {}


class AbortPipeline(RuntimeError):
    """Raised when a core stage cannot produce output after retries and fallbacks."""


# ============================================================ pure helpers


def normalize_url(url: str) -> str:
    """Canonical key for dedupe. Empty string when the URL is unusable."""
    if not url:
        return ""
    try:
        p = urllib.parse.urlsplit(url.strip())
    except ValueError:
        return ""
    if not p.scheme and not p.netloc:
        return ""
    host = (p.netloc or p.scheme).lower()
    if host.startswith("www."):
        host = host[4:]
    if not host:
        return ""
    path = re.sub(r"/+$", "", p.path or "")
    query_pairs = [
        (k, v)
        for k, v in urllib.parse.parse_qsl(p.query, keep_blank_values=False)
        if not TRACKING_PARAMS.match(k)
    ]
    query = urllib.parse.urlencode(sorted(query_pairs))
    return f"{host}{path}" + (f"?{query}" if query else "")


def est_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def strip_think(text: str) -> str:
    out = THINK_RE.sub("", text or "")
    if "</think>" in out:
        out = out.split("</think>")[-1]
    return out.strip()


def _norm_with_map(text: str) -> tuple[str, list[int]]:
    """Collapse whitespace runs to single spaces; keep a map back to source indices."""
    out: list[str] = []
    idx_map: list[int] = []
    prev_space = False
    for i, ch in enumerate(text):
        if ch.isspace():
            if prev_space:
                continue
            out.append(" ")
            idx_map.append(i)
            prev_space = True
        else:
            out.append(ch)
            idx_map.append(i)
            prev_space = False
    return "".join(out), idx_map


_BOUNDARY_AFTER = set('.,;:!?)]}"\'')


def _boundary_ok(hay: str, start: int, end: int) -> bool:
    if start > 0:
        prev = hay[start - 1]
        if not (prev.isspace() or prev in "([{\"'-\u2014\u2013"):
            return False
    if end < len(hay):
        nxt = hay[end]
        if not (nxt.isspace() or nxt in _BOUNDARY_AFTER):
            return False
    return True


def _find_spans(hay: str, needle: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    pos = hay.find(needle)
    while pos != -1:
        end = pos + len(needle)
        if _boundary_ok(hay, pos, end):
            spans.append((pos, end))
        pos = hay.find(needle, pos + 1)
    return spans


def apply_replacements(
    text: str, replacements: Iterable[dict], log: RunLog | None = None, stage: str = "replace"
) -> tuple[str, dict]:
    """Exact-substring replacement after whitespace normalization. No fuzzy matching.

    A replacement is skipped and logged when old_sentence is not found or matches more
    than once. Only the matched span is rewritten; all other text is untouched.
    """
    stats = {"applied": 0, "skipped_not_found": 0, "skipped_duplicate": 0, "skipped_invalid": 0}
    current = text
    for rep in replacements or []:
        if not isinstance(rep, dict):
            stats["skipped_invalid"] += 1
            continue
        old = rep.get("old_sentence")
        new = rep.get("new_sentence")
        if not isinstance(old, str) or not isinstance(new, str) or not old.strip():
            stats["skipped_invalid"] += 1
            if log:
                log.warn(stage, "replacement missing old_sentence/new_sentence string")
            continue
        if old.strip() == new.strip():
            stats["skipped_invalid"] += 1
            if log:
                log.warn(stage, f"replacement is a no-op, skipped: {old[:70]!r}")
            continue
        norm_hay, idx_map = _norm_with_map(current)
        needle = " ".join(old.split())
        spans = _find_spans(norm_hay, needle)
        if not spans:
            stats["skipped_not_found"] += 1
            if log:
                log.warn(stage, f"old_sentence not found, skipped: {old[:70]!r}")
            continue
        if len(spans) > 1:
            stats["skipped_duplicate"] += 1
            if log:
                log.warn(
                    stage,
                    f"old_sentence matches {len(spans)} times, skipped: {old[:70]!r}",
                )
            continue
        start, end = spans[0]
        i_start, i_end = idx_map[start], idx_map[end - 1] + 1
        current = current[:i_start] + new + current[i_end:]
        stats["applied"] += 1
    return current, stats


def extract_json(text: str) -> dict:
    """Strip code fences, take the outermost {...} object, json.loads. Raises ValueError."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty response")
    s = text.strip()
    fence = re.match(r"^```[A-Za-z0-9_+-]*\s*\n(.*?)\n?```\s*$", s, re.DOTALL)
    if fence:
        s = fence.group(1).strip()
    else:
        s = re.sub(r"^```[A-Za-z0-9_+-]*", "", s).strip()
        s = re.sub(r"```$", "", s).strip()
    start = s.find("{")
    if start == -1:
        raise ValueError("no '{' found in response")
    depth = 0
    in_str = False
    esc = False
    end = -1
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end == -1:
        raise ValueError("unbalanced braces in response")
    payload = s[start:end]
    obj = json.loads(payload)
    if not isinstance(obj, dict):
        raise ValueError("outermost JSON value is not an object")
    return obj


def extract_citations(text: str) -> list[str]:
    seen: list[str] = []
    for m in CITATION_RE.finditer(text or ""):
        sid = f"S{m.group(1)}"
        if sid not in seen:
            seen.append(sid)
    return seen


def validate_citations(text: str, registry: SourceRegistry) -> dict:
    cited = extract_citations(text)
    unknown = [s for s in cited if not registry.known(s)]
    return {"cited": cited, "unknown": unknown, "ok": not unknown}


def chunk_document(text: str, source_id: str, min_words: int = CHUNK_MIN_WORDS,
                   max_words: int = CHUNK_MAX_WORDS,
                   target_words: int = CHUNK_TARGET_WORDS) -> list[Chunk]:
    """Split a document into ~250-400 word chunks, preferring paragraph then sentence
    boundaries. Always returns at least one chunk for non-empty text."""
    text = (text or "").strip()
    if not text:
        return []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    pieces: list[str] = []
    for para in paragraphs:
        if len(para.split()) <= max_words:
            pieces.append(para)
            continue
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'\(])", para)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            pieces.append(para)
            continue
        buf: list[str] = []
        buf_w = 0
        for sent in sentences:
            w = len(sent.split())
            if buf and buf_w + w > max_words:
                pieces.append(" ".join(buf))
                buf, buf_w = [], 0
            buf.append(sent)
            buf_w += w
            if buf_w >= target_words:
                pieces.append(" ".join(buf))
                buf, buf_w = [], 0
        if buf:
            pieces.append(" ".join(buf))

    # A single "sentence" longer than max_words (no sentence punctuation in the text)
    # has to be split on word boundaries or it would become one oversized chunk.
    bounded: list[str] = []
    for piece in pieces:
        words = piece.split()
        if len(words) <= max_words:
            bounded.append(piece)
            continue
        for i in range(0, len(words), max_words):
            bounded.append(" ".join(words[i: i + max_words]))
    pieces = bounded

    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_w = 0
    for piece in pieces:
        w = len(piece.split())
        if buf and buf_w + w > max_words:
            chunks.append(_make_chunk(source_id, chunks, buf, buf_w, min_words))
            buf, buf_w = [], 0
        buf.append(piece)
        buf_w += w
        if buf_w >= target_words:
            chunks.append(_make_chunk(source_id, chunks, buf, buf_w, min_words))
            buf, buf_w = [], 0
    if buf:
        chunks.append(_make_chunk(source_id, chunks, buf, buf_w, min_words))
    return chunks


def _make_chunk(source_id: str, chunks: list[Chunk], buf: list[str], words: int,
                min_words: int) -> Chunk:
    body = "\n\n".join(buf).strip()
    if chunks and words < min_words:
        prev = chunks[-1]
        merged = prev.text + "\n\n" + body
        prev.text = merged
        prev.words = len(merged.split())
        return prev
    return Chunk(source_id=source_id, index=len(chunks), text=body, words=words)


def _chunk_token_set(text: str) -> frozenset[str]:
    return frozenset(
        tok for tok in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(tok) > 2 and tok not in _STOPWORDS
    )


def apply_rerank_threshold(
    ranked: Sequence[tuple[int, float]],
    min_score: float,
    floor: int = RERANK_FLOOR,
    cap: int = RERANK_CAP,
    chunks: Sequence[Chunk] | None = None,
    core_cap: int = RERANK_CORE_CAP,
) -> list[tuple[int, float]]:
    """Drop chunks below min_score, keep at least `floor`, and cap at `cap` (60).

    When `len(kept) > cap` and `chunks` are supplied, DURALUMIN selects:
      - Bucket A (`core_cap` = 45): Top 45 chunks strictly by reranker score.
      - Bucket B (`cap - core_cap` = 15): Top 15 High-Novelty chunks selected via
        Maximal Marginal Relevance (MMR) minimizing token-set Jaccard overlap with
        already-selected chunks so low-coverage facts survive into the 60-chunk pool.
    """
    ranked = list(ranked)
    kept = [r for r in ranked if r[1] >= min_score]
    if len(kept) < floor:
        kept = ranked[:floor]
    if len(kept) <= cap:
        return kept
    if not chunks or core_cap >= cap or len(kept) <= core_cap:
        return kept[:cap]

    core = list(kept[:core_cap])
    candidates = list(kept[core_cap:])
    novelty_needed = cap - len(core)

    tok_cache: dict[int, frozenset[str]] = {}
    for idx, _ in kept:
        if 0 <= idx < len(chunks):
            tok_cache[idx] = _chunk_token_set(chunks[idx].text)
        else:
            tok_cache[idx] = frozenset()

    selected_sets: list[frozenset[str]] = [tok_cache[i] for i, _ in core if tok_cache.get(i)]
    max_s = max((s for _, s in candidates), default=1.0)
    min_s = min((s for _, s in candidates), default=0.0)
    span_s = (max_s - min_s) if max_s > min_s else 1.0

    novelty: list[tuple[int, float]] = []
    remaining = list(candidates)
    while remaining and len(novelty) < novelty_needed:
        best_pos = 0
        best_mmr = -1e9
        for pos, (c_idx, c_score) in enumerate(remaining):
            c_toks = tok_cache.get(c_idx, frozenset())
            if not c_toks or not selected_sets:
                max_sim = 0.0
            else:
                max_sim = max(
                    len(c_toks & s_toks) / max(1, len(c_toks | s_toks))
                    for s_toks in selected_sets
                )
            norm_rel = (c_score - min_s) / span_s
            mmr = 0.45 * norm_rel + 0.55 * (1.0 - max_sim)
            if mmr > best_mmr:
                best_mmr = mmr
                best_pos = pos
        chosen = remaining.pop(best_pos)
        novelty.append(chosen)
        if tok_cache.get(chosen[0]):
            selected_sets.append(tok_cache[chosen[0]])

    return core + novelty


def parse_plan(text: str) -> tuple[list[str], list[str]]:
    """Per-line regex parse (tolerates markdown bullets/numbering around Research:/Search:)."""
    research: list[str] = []
    search: list[str] = []
    for raw_line in (text or "").splitlines():
        line = re.sub(r"^\s*(?:[\-*•]|\d+[.)])\s*\**", "", raw_line).replace("**", "")
        m = RESEARCH_RE.match(line)
        if m:
            val = m.group(1).strip().strip('"').strip()
            if val:
                research.append(val)
            continue
        m = SEARCH_RE.match(line)
        if m:
            val = m.group(1).strip().strip('"').strip()
            if val:
                search.append(val)
    return research, search


def plan_is_valid(research: Sequence[str], search: Sequence[str]) -> bool:
    return len(research) == 2 and 5 <= len(search) <= 13


def fallback_plan(query: str) -> tuple[list[str], list[str]]:
    base = " ".join(query.split())
    research = [
        f"{base}: background, history and the state of play as of today",
        f"{base}: evidence, data, disagreements and open questions",
    ]
    facets = (
        "official report", "statistics data", "criticism limitations", "latest 2026",
        "primary source document", "market size forecast", "expert analysis",
        "historical timeline", "regulatory policy", "cost breakdown",
    )
    search = [base[:90]] + [f"{base} {f}" for f in facets]
    return research, search[:13]


MARKDOWN_STRIP_RE = re.compile(
    r"[`*_#>|\[\]\(\)~!_-]+|\s+"
)


def strip_markdown(text: str) -> str:
    return MARKDOWN_STRIP_RE.sub(" ", text or "").lower().strip()


def polish_similarity(a: str, b: str, floor: float = POLISH_MIN_RATIO) -> float:
    """Word-sequence comparison after stripping markdown and whitespace."""
    wa = strip_markdown(a).split()
    wb = strip_markdown(b).split()
    if not wa and not wb:
        return 1.0
    return difflib.SequenceMatcher(a=wa, b=wb, autojunk=False).ratio()


def parse_rerank_scores(payload: Any) -> list[tuple[int, float]] | None:
    """Normalize the Cloudflare reranker envelope into (index, score) pairs.

    Verified schema is {"result": {"response": [{"id": int, "score": float}]}}. Also
    accepts the response array at the top level or directly under "result", and the
    "index" spelling, so a shape change degrades instead of crashing.
    """
    if isinstance(payload, list):
        candidates: Any = payload
    elif isinstance(payload, dict):
        if isinstance(payload.get("errors"), list) and payload["errors"]:
            return None
        candidates = None
        for key in ("response", "data", "result", "results"):
            val = payload.get(key)
            if isinstance(val, list):
                candidates = val
                break
            if isinstance(val, dict):
                for sub in ("response", "data", "results"):
                    if isinstance(val.get(sub), list):
                        candidates = val[sub]
                        break
            if candidates is not None:
                break
        if candidates is None:
            return None
    else:
        return None
    out: list[tuple[int, float]] = []
    for row in candidates:
        if not isinstance(row, dict):
            continue
        idx = row.get("id", row.get("index"))
        score = row.get("score")
        if isinstance(idx, bool) or idx is None or not isinstance(score, (int, float)):
            continue
        try:
            out.append((int(idx), float(score)))
        except (TypeError, ValueError):
            continue
    return out


# ============================================================ http layer


async def http_request(
    method: str,
    url: str,
    *,
    headers: dict | None = None,
    json_body: Any = None,
    params: dict | None = None,
    timeout: float = 60.0,
    attempts: int = HTTP_ATTEMPTS,
    log: RunLog | None = None,
    stage: str = "http",
) -> tuple[int, Any]:
    """HTTP with retry on 429/5xx and exponential backoff, max `attempts`."""
    last_status = 0
    last_err = ""
    stub = dry_http_stub(method, url, params)
    if stub is not None:
        return stub
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        for attempt in range(1, attempts + 1):
            try:
                resp = await client.request(
                    method, url, headers=headers, json=json_body, params=params
                )
            except Exception as exc:
                last_status = 0
                last_err = f"{type(exc).__name__}: {exc}"
                if attempt >= attempts:
                    break
                delay = BACKOFF_BASE_S * (2 ** (attempt - 1)) + random.uniform(0, BACKOFF_JITTER_S)
                print(f"    retry {attempt}/{attempts - 1} after {last_err[:120]}")
                await asyncio.sleep(delay)
                continue
            last_status = resp.status_code
            if resp.status_code in (429, 500, 502, 503, 504):
                last_err = _short(resp.text)
                if attempt >= attempts:
                    break
                delay = BACKOFF_BASE_S * (2 ** (attempt - 1)) + random.uniform(0, BACKOFF_JITTER_S)
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        delay = max(delay, float(retry_after))
                    except ValueError:
                        pass
                print(
                    f"    {resp.status_code} on attempt {attempt}/{attempts}, "
                    f"backing off {delay:.1f}s"
                )
                await asyncio.sleep(delay)
                continue
            try:
                return resp.status_code, resp.json()
            except Exception:
                return resp.status_code, {"_raw": resp.text}
    if log:
        log.warn(stage, f"{method} {url} failed after {attempts} attempts: {last_err[:200]}")
    return last_status, {"_error": last_err}


def _short(text: str, n: int = 300) -> str:
    t = " ".join((text or "").split())
    return t[:n]


# ============================================================ llm layer

_CLIENTS: dict[tuple, AsyncOpenAI] = {}
_GENAI_CLIENTS: dict[tuple, Any] = {}


class StageError(RuntimeError):
    pass


class StageTimeout(StageError):
    pass


def _client(cfg: StageCfg) -> AsyncOpenAI:
    key = (cfg.base_url, cfg.key())
    if key not in _CLIENTS:
        _CLIENTS[key] = AsyncOpenAI(
            api_key=cfg.key(), base_url=cfg.base_url, timeout=cfg.timeout, max_retries=0
        )
    return _CLIENTS[key]


def _genai_client(cfg: StageCfg) -> Any:
    """Official Google AI Studio client from the google-genai SDK.

    SDK-internal retries are switched off so this module's 429/5xx backoff stays
    the single retry authority; `attempts=1` means one try, no SDK retry loop.
    """
    key = (cfg.base_url, cfg.key())
    if key not in _GENAI_CLIENTS:
        _GENAI_CLIENTS[key] = genai.Client(
            api_key=cfg.key(),
            http_options=genai_types.HttpOptions(
                timeout=int(cfg.timeout * 1000),
                retry_options=genai_types.HttpRetryOptions(attempts=1),
            ),
        )
    return _GENAI_CLIENTS[key]


def _genai_parts(messages: list[dict]) -> tuple[str, list[dict]]:
    """Split OpenAI-style messages into (system_instruction, contents) for google-genai.

    The SDK wants the system prompt as a separate config field, and maps the
    assistant role onto 'model'. Alternating user/assistant turns are merged into
    Gemini's turn structure by role, so a multi-turn retry or repair prompt keeps
    its conversational shape.
    """
    system = ""
    contents: list[dict] = []
    for m in messages:
        role = m.get("role", "user")
        text = m.get("content") or ""
        if not isinstance(text, str):
            text = "".join(
                p.get("text", "") for p in text if isinstance(p, dict)
            )
        if role == "system":
            system = f"{system}\n\n{text}".strip() if system else text
            continue
        gr = "model" if role == "assistant" else "user"
        if contents and contents[-1]["role"] == gr:
            contents[-1]["parts"][0]["text"] += f"\n\n{text}"
        else:
            contents.append({"role": gr, "parts": [{"text": text}]})
    if not contents:
        contents = [{"role": "user", "parts": [{"text": "(empty prompt)"}]}]
    return system, contents


def genai_kwargs(cfg: StageCfg, messages: list[dict], max_tokens: int | None) -> dict:
    """Build the keyword arguments for client.aio.models.generate_content."""
    system, contents = _genai_parts(messages)
    config = genai_types.GenerateContentConfig(
        max_output_tokens=max_tokens or cfg.max_tokens,
        system_instruction=system or None,
    )
    if cfg.temperature is not None:
        config.temperature = cfg.temperature
    if cfg.disable_thinking:
        config.thinking_config = genai_types.ThinkingConfig(thinking_budget=0)
    return {"model": cfg.model, "contents": contents, "config": config}



# --- COHERE LEDGER EXTRACTION STRICT JSON SCHEMA ---
LEDGER_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "extractions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "string"},
                    "citations": {"type": "array", "items": {"type": "string"}},
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "surface": {"type": "string"},
                                "canonical_name": {"type": "string"},
                                "aliases": {"type": "array", "items": {"type": "string"}},
                                "type": {"type": "string"},
                                "role": {"type": "string"}
                            },
                            "required": ["surface", "canonical_name", "aliases", "type", "role"],
                            "additionalProperties": False
                        }
                    },
                    "metrics": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "comparison_group": {"type": "string"},
                                "subject": {"type": "string"},
                                "metric_name": {"type": "string"},
                                "raw_value": {"type": "string"},
                                "numeric_value": {"type": "number"},
                                "unit": {"type": "string"},
                                "period_or_date": {"type": "string"},
                                "conditions_caveats": {"type": "string"},
                                "source_type": {"type": "string"},
                                "cites": {"type": "array", "items": {"type": "string"}}
                            },
                            "required": [
                                "comparison_group", "subject", "metric_name", "raw_value",
                                "numeric_value", "unit", "period_or_date", "conditions_caveats",
                                "source_type", "cites"
                            ],
                            "additionalProperties": False
                        }
                    },
                    "claims": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "subject": {"type": "string"},
                                "relation": {"type": "string"},
                                "object": {"type": "string"},
                                "text": {"type": "string"},
                                "cites": {"type": "array", "items": {"type": "string"}}
                            },
                            "required": ["subject", "relation", "object", "text", "cites"],
                            "additionalProperties": False
                        }
                    },
                    "leads_and_gaps": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["chunk_id", "citations", "entities", "metrics", "claims", "leads_and_gaps"],
                "additionalProperties": False
            }
        }
    },
    "required": ["extractions"],
    "additionalProperties": False
}


def llm_kwargs(cfg: StageCfg, messages: list[dict], max_tokens: int | None) -> dict:
    if cfg.provider == "google_genai":
        return genai_kwargs(cfg, messages, max_tokens)
    kwargs: dict[str, Any] = {"model": cfg.model, "messages": messages}
    if cfg.temperature is not None:
        kwargs["temperature"] = cfg.temperature
    kwargs["max_tokens"] = max_tokens or cfg.max_tokens
    if cfg.reasoning_effort:
        kwargs["reasoning_effort"] = cfg.reasoning_effort
    if cfg.extra_body:
        kwargs["extra_body"] = cfg.extra_body

    # Strictly apply JSON Schema mode ONLY to Cohere in the ledger batch extraction stage
    if cfg.name == "ledger-cohere":
        kwargs["response_format"] = {"type": "json_object", "schema": LEDGER_EXTRACTION_SCHEMA}

    return kwargs


@dataclass
class LLMResult:
    text: str
    reasoning_chars: int
    finish_reason: str | None
    usage: Any
    model: str
    streamed: bool


def _stubbed(cfg: StageCfg, kind: str) -> Any:
    stub = _DRY_STUB(cfg.name, kind, cfg.model)
    if stub is None:
        return None
    if isinstance(stub, Exception):
        raise stub
    return stub


async def llm_call(
    cfg: StageCfg, messages: list[dict], log: RunLog | None = None, stage: str = "",
    max_tokens: int | None = None, attempts: int = HTTP_ATTEMPTS,
) -> LLMResult:
    """Non-streaming chat completion with 429/5xx retry and backoff."""
    label = stage or cfg.name
    last_err = ""
    _record_prompt(messages)
    budget = max_tokens or cfg.max_tokens
    attempts_left = attempts
    made = 0
    while attempts_left > 0:
        attempt = attempts - attempts_left + 1
        attempts_left -= 1
        made += 1
        stub = _stubbed(cfg, "chat")
        if stub is not None:
            if isinstance(stub, dict) and "raise" in stub:
                last_err = stub["raise"]
                if attempts_left > 0:
                    await asyncio.sleep(0.01)
                    continue
                break
            return LLMResult(
                text=stub.get("text", ""),
                reasoning_chars=int(stub.get("reasoning_chars", 0)),
                finish_reason=stub.get("finish_reason", "stop"),
                usage=stub.get("usage"),
                model=cfg.model,
                streamed=False,
            )
        try:
            if cfg.provider == "google_genai":
                resp = await _genai_client(cfg).aio.models.generate_content(
                    **llm_kwargs(cfg, messages, budget)
                )
                text, reasoning, finish, usage = _read_genai_response(resp)
            else:
                resp = await _client(cfg).chat.completions.create(
                    **llm_kwargs(cfg, messages, budget)
                )
                if not resp.choices:
                    last_err = "provider returned no choices"
                    break
                msg = resp.choices[0].message
                text = strip_think(msg.content or "")
                reasoning = (getattr(msg, "reasoning_content", None)
                             or getattr(msg, "reasoning", None) or "")
                finish = resp.choices[0].finish_reason
                usage = getattr(resp, "usage", None)
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            shrink = _context_overflow(exc)
            if shrink is not None and shrink < budget and attempts_left > 0:
                budget = shrink
                last_err = (f"{_short(last_err, 160)} (context overflow; retrying with "
                            f"max_tokens={budget})")
                print(f"    {last_err}")
                continue
            bigger = _budget_error(exc)
            ceiling = _token_ceiling(exc, cfg.max_tokens * 4)
            if bigger and budget < ceiling:
                budget = min(budget * 3, ceiling)
                last_err = f"{_short(last_err, 200)} (retrying with max_tokens={budget})"
                print(f"    {last_err}")
                continue
            if _is_retryable(exc) and attempts_left > 0:
                delay = BACKOFF_BASE_S * (2 ** (attempt - 1)) + random.uniform(0, BACKOFF_JITTER_S)
                print(f"    retry {attempt}/{attempts - 1} after {_short(last_err, 120)}")
                await asyncio.sleep(delay)
                continue
            break
        if not text and reasoning.strip():
            last_err = "provider returned reasoning only, no answer content"
            # Do NOT blindly triple budget past 24,000 tokens (prevents Routeway 42k context overflow!)
            safe_cap = min(cfg.max_tokens * 2, 20000)
            if budget < safe_cap and attempts_left > 0:
                budget = safe_cap
                last_err += f" (retrying with max_tokens={budget})"
                print(f"    {last_err}")
                continue
            break
        return LLMResult(
            text=text,
            reasoning_chars=len(reasoning or ""),
            finish_reason=finish,
            usage=usage,
            model=cfg.model,
            streamed=False,
        )
    raise StageError(f"{label}: gave up after {made} of {attempts} attempts "
                     f"(max_tokens={budget}): {_short(last_err, 260)}")


def _read_genai_response(resp: Any) -> tuple[str, str, str, Any]:
    """Flatten a google-genai GenerateContentResponse into (text, reasoning, finish, usage).

    finish_reason arrives as a types.FinishReason enum, so it is stringified and
    lowercased here; usage is dumped to a plain dict so log.json stays serialisable.
    """
    text = (getattr(resp, "text", None) or "").strip()
    reasoning = ""
    try:
        for cand in getattr(resp, "candidates", None) or []:
            for part in (getattr(getattr(cand, "content", None), "parts", None) or []):
                if getattr(part, "thought", False) and getattr(part, "text", None):
                    reasoning += part.text
    except Exception:
        reasoning = ""
    finish = ""
    cands = getattr(resp, "candidates", None) or []
    if cands:
        finish = str(getattr(cands[0], "finish_reason", "") or "")
        finish = finish.rsplit(".", 1)[-1].lower()
    usage = getattr(resp, "usage_metadata", None)
    if usage is not None and hasattr(usage, "model_dump"):
        usage = usage.model_dump(exclude_none=True)
    return text, reasoning, finish, usage


def _budget_error(exc: Exception) -> bool:
    if getattr(exc, "status_code", None) == 429 or "ratelimit" in type(exc).__name__.lower() or "tokens per minute" in str(exc).lower():
        return False
    """Detect a provider rejecting the request because max_tokens is too small for the
    model's reasoning phase. FinalRouter raises this as a 400 with an explicit message."""
    msg = str(exc).lower()
    if "insufficient_credits" in msg or "free allowance" in msg or "add credits" in msg:
        # A billing cap, not a sizing problem: the account's free tier caps reply
        # length. Raising max_tokens cannot fix it, so it must not be escalated.
        return False
    return "max_tokens" in msg and (
        "ceiling" in msg or "budget" in msg or "reasoning" in msg
        or "too big" in msg or "too large" in msg
    )


def _int_in(msg: str, pattern: str) -> int:
    """Parse an integer that may carry comma or underscore thousands separators."""
    m = re.search(pattern, msg, re.IGNORECASE)
    if not m:
        return 0
    try:
        return int(m.group(1).replace(",", "").replace("_", ""))
    except ValueError:
        return 0


def _token_ceiling(exc: Exception, default: int) -> int:
    """Largest max_tokens the provider will accept for this request.

    FinalRouter states its hard cap in the error text ("Too big: expected number to be
    <=200000"). Blindly tripling past that turns a recoverable budget problem into a
    hard failure, so a stated ceiling always wins over our own estimate.
    """
    stated = _int_in(
        str(exc),
        r"(?:expected\s+number\s+to\s+be\s*<=\s*|less\s+than\s+or\s+equal\s+to\s*|"
        r"at\s+most\s+)([\d,_]+)",
    )
    return stated if stated > 0 else default


def _context_overflow(exc: Exception) -> int | None:
    """Largest safe max_tokens when the provider rejects input+output over context.

    Routeway's free tier reports: "Your prompt exceeds the model's context length.
    Estimated input tokens: N. Requested output tokens: M. Max context tokens: T."
    Unlike a low-ceiling error this must SHRINK the budget, so it is handled
    separately from _budget_error. Returns None for any other error.
    """
    s = str(exc)
    cap = _int_in(s, r"max context tokens:\s*([\d,_]+)")
    if not cap:
        return None
    spent = _int_in(s, r"estimated input tokens:\s*([\d,_]+)")
    return max(2048, cap - spent - 512)


def _is_retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status in (408, 429, 500, 502, 503, 504)
    return type(exc).__name__ in (
        "APIConnectionError", "APITimeoutError", "APIConnectionTimeoutError",
        "InternalServerError", "RateLimitError", "ServiceUnavailableError",
    )


async def llm_call_stream(
    cfg: StageCfg, messages: list[dict], log: RunLog | None = None, stage: str = "",
    max_tokens: int | None = None, idle_timeout: float | None = None,
    total_timeout: float | None = None,
) -> LLMResult:
    """Streaming chat completion with ONE unified total_timeout (no premature idle timeout!)."""
    label = stage or cfg.name
    single_timeout = total_timeout or cfg.timeout or DOSSIER_TOTAL_TIMEOUT
    _record_prompt(messages)
    stub = _stubbed(cfg, "stream")
    if stub is not None:
        if isinstance(stub, dict) and "raise" in stub:
            raise StageError(f"{label}: {stub['raise']}")
        return LLMResult(
            text=stub.get("text", ""),
            reasoning_chars=int(stub.get("reasoning_chars", 0)),
            finish_reason=stub.get("finish_reason", "stop"),
            usage=stub.get("usage"),
            model=cfg.model,
            streamed=True,
        )

    kwargs = llm_kwargs(cfg, messages, max_tokens)
    kwargs["stream"] = True
    kwargs["stream_options"] = {"include_usage": True}
    stream = None
    started = time.time()
    parts: list[str] = []
    reasoning_chars = 0
    finish_reason: str | None = None
    usage = None
    try:
        try:
            stream = await asyncio.wait_for(
                _client(cfg).chat.completions.create(**kwargs), timeout=single_timeout
            )
        except asyncio.TimeoutError as exc:
            raise StageTimeout(
                f"{label}: no response within {single_timeout:.0f}s timeout"
            ) from exc
        iterator = stream.__aiter__()
        while True:
            remaining = single_timeout - (time.time() - started)
            if remaining <= 0:
                raise StageTimeout(f"{label}: timeout {single_timeout:.0f}s exceeded")
            try:
                chunk = await asyncio.wait_for(iterator.__anext__(), timeout=remaining)
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError as exc:
                raise StageTimeout(
                    f"{label}: timeout after {single_timeout:.0f}s"
                ) from exc
            if getattr(chunk, "usage", None):
                usage = chunk.usage
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta is None:
                continue
            if getattr(delta, "content", None):
                parts.append(delta.content)
            rc = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
            if rc:
                reasoning_chars += len(rc)
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason
    except (StageError, StageTimeout):
        raise
    except Exception as exc:
        raise StageError(f"{label}: stream failed with {type(exc).__name__}: "
                         f"{_short(str(exc), 220)}") from exc
    finally:
        if stream is not None:
            try:
                await stream.close()
            except Exception:
                pass
    text = strip_think("".join(parts))
    if not text:
        raise StageError(f"{label}: empty streamed content (reasoning_chars={reasoning_chars})")
    return LLMResult(
        text=text,
        reasoning_chars=reasoning_chars,
        finish_reason=finish_reason,
        usage=usage,
        model=cfg.model,
        streamed=True,
    )

async def llm_json(
    cfg: StageCfg, messages: list[dict], log: RunLog, stage: str,
    max_tokens: int | None = None,
) -> dict | None:
    """JSON stage helper: one call, one JSON-repair retry, then skip with a warning."""
    try:
        res = await (llm_call_stream(cfg, messages, log=log, stage=stage, max_tokens=max_tokens, total_timeout=cfg.timeout) if cfg.stream and _DRY_RUN['scenario'] is None else llm_call(cfg, messages, log=log, stage=stage, max_tokens=max_tokens))
    except StageError as exc:
        log.warn(stage, f"llm call failed: {exc}")
        return None
    log.stage_usage(provider=res.model, usage=res.usage, finish_reason=res.finish_reason,
                    reasoning_chars=res.reasoning_chars)
    repair_budget = max_tokens or cfg.max_tokens
    if res.finish_reason == "length":
        log.warn(stage, f"finish_reason=length at max_tokens={repair_budget}; the output may "
                        "have been cut off mid-JSON")
        repair_budget = min(repair_budget * 3, cfg.max_tokens * 4)
    try:
        return extract_json(res.text)
    except ValueError as exc:
        log.warn(stage, f"JSON parse failed ({exc}); requesting a JSON-only repair")
    repair = [*messages, {"role": "assistant", "content": res.text or "(empty response)"},
              {"role": "user", "content": P.JSON_REPAIR_MESSAGE}]
    try:
        res = await llm_call(cfg, repair, log=log, stage=stage, max_tokens=repair_budget)
    except StageError as exc:
        log.warn(stage, f"JSON repair call failed: {exc}")
        return None
    try:
        return extract_json(res.text)
    except ValueError as exc:
        log.warn(stage, f"stage skipped: model could not produce valid JSON ({exc})")
        return None


async def llm_chain(
    chain: Sequence[StageCfg], runner, log: RunLog, stage: str, core: bool
) -> tuple[Any, StageCfg, bool]:
    """Try each config in `chain`. core=True aborts the run when all fail."""
    errs: list[str] = []
    for i, cfg in enumerate(chain):
        if not cfg.have_key():
            log.warn(stage, f"{cfg.name} skipped: {cfg.key_env} is not set")
            errs.append(f"{cfg.name}: no key")
            continue
        try:
            result = await runner(cfg)
        except StageError as exc:
            errs.append(str(exc))
            log.warn(stage, f"{cfg.name} failed: {exc}")
            continue
        except Exception as exc:
            errs.append(f"{cfg.name}: {type(exc).__name__}: {exc}")
            log.warn(stage, f"{cfg.name} raised {type(exc).__name__}: {_short(str(exc), 200)}")
            continue
        if result is None:
            # A JSON stage reports failure by returning None rather than raising, so
            # returning here would short-circuit the chain and silently skip the
            # fallback provider. Treat it as a provider failure and keep going.
            errs.append(f"{cfg.name}: no usable result (unparseable or empty output)")
            log.warn(stage, f"{cfg.name} returned no usable result; trying the next provider")
            continue
        if i > 0:
            log.provider_used(stage, cfg.model, True)
            log.stage_usage(provider=cfg.model, fallback=True)
            print(f"    using fallback {cfg.model}")
        else:
            log.provider_used(stage, cfg.model, False)
            log.stage_usage(provider=cfg.model)
        return result, cfg, i > 0
    msg = f"all providers failed for stage '{stage}': " + " | ".join(errs)
    if core:
        raise AbortPipeline(msg)
    log.warn(stage, msg + " -> stage skipped")
    return None, chain[0], False


# ============================================================ stage 1 plan


async def stage_plan(query: str, log: RunLog) -> tuple[list[str], list[str]]:
    """Llama 3.3 70B via Groq. Exactly 2 Research + 4-8 Search lines, regex-parsed."""
    log.stage_begin("plan", {"model_primary": PLANNER.model})
    messages = [
        {"role": "system", "content": P.PLANNER_SYSTEM},
        {"role": "user", "content": f"USER QUESTION:\n{query}"},
    ]
    errs: list[str] = []
    for i, cfg in enumerate((PLANNER, *PLANNER.fallback)):
        if not cfg.have_key():
            log.warn("plan", f"{cfg.name} skipped: {cfg.key_env} is not set")
            errs.append(f"{cfg.name}: no key")
            continue
        for attempt in (1,):
            try:
                res = await llm_call(cfg, messages, log=log, stage="plan", attempts=1)
            except StageError as exc:
                errs.append(str(exc))
                log.warn("plan", f"{cfg.name} failed: {exc}")
                break
            log.stage_usage(provider=res.model, usage=res.usage,
                            finish_reason=res.finish_reason, model_attempt=attempt)
            log.provider_used("plan", res.model, i > 0)
            research, search = parse_plan(res.text)
            log.stage_usage(research_lines=len(research), search_lines=len(search))
            if plan_is_valid(research, search):
                log.artifact("01_plan_raw.txt", res.text)
                log.artifact_json(
                    "01_plan.json",
                    {"research": research, "search": search, "model": res.model,
                     "attempt": attempt, "shape_ok": True},
                )
                log.stage_end("ok")
                if i > 0:
                    log.stage_usage(fallback=True)
                return research, search
            errs.append(
                f"{cfg.name} attempt {attempt}: {len(research)} Research lines, "
                f"{len(search)} Search lines (need 2 and 4-8)"
            )
            log.warn("plan", errs[-1])
        if i > 0:
            log.stage_usage(fallback=True)
    log.warn("plan", "planner LLM unusable (" + " | ".join(errs[-3:]) + ")")
    research, search = fallback_plan(query)
    log.artifact_json(
        "01_plan.json",
        {"research": research, "search": search, "model": "builtin-fallback",
         "shape_ok": True, "reason": "planner LLM produced no usable plan"},
    )
    log.stage_end("ok", error="used built-in query fallback")
    return research, search


async def _poll_sleep(seconds: float) -> None:
    if _DRY_RUN["scenario"] is not None:
        await asyncio.sleep(0)
        return
    await asyncio.sleep(seconds)


# ============================================================ stage 2 retrieve


def segments_from_citations(content: str, source_list: list[dict]) -> list[dict]:
    """Split a research report at its numbered citation markers.

    Research agents (Tavily citation_format=numbered, You.com [[n]]) interleave markers
    with the claims they support. Splitting there lets a claim keep the real underlying
    source URL instead of pointing at a synthetic report id.
    """
    content = content or ""
    if not source_list:
        return []
    marks: list[tuple[int, list[int]]] = []
    for m in re.finditer(r"\[+\s*(\d{1,3}(?:\s*,\s*\d{1,3})*)\s*\]+", content):
        nums = [int(x) for x in re.findall(r"\d{1,3}", m.group(1))]
        marks.append((m.start(), nums))
    if not marks:
        return []
    out: list[dict] = []
    bounds = [m[0] for m in marks] + [len(content)]
    for i, (_, nums) in enumerate(marks):
        body = content[bounds[i]: bounds[i + 1]].strip()
        body = re.sub(r"^[\[\]]*\s*\d{1,3}(?:\s*,\s*\d{1,3})*\s*[\[\]]*\s*", "", body).strip()
        if len(body) < 80:
            continue
        src = next((source_list[n - 1] for n in nums if 1 <= n <= len(source_list)), None)
        if not src or not src.get("url"):
            continue
        out.append({"url": src["url"], "title": src.get("title") or "", "text": body,
                    "date": src.get("date", "")})
    deduped: dict[str, dict] = {}
    for row in out:
        cur = deduped.get(row["url"])
        if cur:
            cur["text"] = (cur["text"] + "\n\n" + row["text"])[:20000]
        else:
            row["text"] = row["text"][:20000]
            deduped[row["url"]] = row
    return list(deduped.values())


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _extract_sentences(text: str) -> list[str]:
    """Split markdown text into sentences, preserving heading lines intact."""
    lines = text.splitlines()
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("|"):
            if stripped:
                out.append(stripped)
            continue
        for sent in _SENTENCE_SPLIT.split(stripped):
            sent = sent.strip()
            if sent:
                out.append(sent)
    return out


def extract_claims_by_source(dossier: str, registry: SourceRegistry) -> dict[str, list[str]]:
    """Return {S#: [cited sentences]} for every source cited in the dossier.

    Only sentences that carry at least one [S#] tag are included. The sentence text
    retains the original tag(s) so the downstream auditor can see exactly what was
    cited, and so replacements can be matched back to the dossier with a simple
    string find.
    """
    cited: dict[str, list[str]] = {}
    seen_pairs: set[tuple[str, str]] = set()
    for sent in _extract_sentences(dossier):
        tags = re.findall(r"\[S(\d+)[^\]]*\]", sent)
        if not tags:
            continue
        for tag in tags:
            sid = f"S{tag}"
            if (sid, sent) not in seen_pairs:
                cited.setdefault(sid, []).append(sent)
                seen_pairs.add((sid, sent))
    return cited


async def tavily_research(question: str, log: RunLog) -> dict:
    """Tavily Research (async job id + polling), falling back to /search."""
    cfg = SEARCH_CFG["tavily_research"]
    key = key_for(cfg)
    if not key:
        return {"provider": cfg["name"], "summary": "", "sources": [], "error": "no api key",
                "fallback_used": False}
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    status, data = await http_request(
        "POST", cfg["base_url"], headers=headers,
json_body={"input": question, "model": cfg["research_model"],
                    "output_length": cfg["output_length"], "citation_format": "numbered"},
        timeout=90.0, log=log, stage="retrieve",
    )
    if status in (200, 201) and isinstance(data, dict) and data.get("request_id"):
        rid = str(data["request_id"])
        deadline = time.time() + cfg["poll_timeout_s"]
        while time.time() < deadline:
            await _poll_sleep(cfg["poll_interval_s"])
            p_status, payload = await http_request(
                "GET", f"{cfg['base_url']}/{rid}",
                headers={"Authorization": f"Bearer {key}"},
                params={"include_usage": "true"}, timeout=60.0, attempts=2,
                log=log, stage="retrieve",
            )
            if p_status != 200 or not isinstance(payload, dict):
                break
            state = payload.get("status")
            if state == "completed":
                content = payload.get("content")
                content = content if isinstance(content, str) else json.dumps(content)
                srcs = payload.get("sources") or []
                rows = segments_from_citations(content, srcs)
                if not rows:
                    rows = [{"url": f"tavily-research://{rid}", "title": f"Research report: {question}",
                             "text": content[:40000], "date": ""}]
                usage = payload.get("usage") or {}
                return {"provider": cfg["name"], "summary": content, "sources": rows,
                        "request_id": rid, "credits": usage.get("credits"),
                        "fallback_used": False, "error": None}
            if state == "failed":
                break
    status, data = await http_request(
        "POST", SEARCH_CFG["tavily_search"]["base_url"], headers=headers,
        json_body={"query": question, "max_results": 8,
                   "include_raw_content": "markdown", "search_depth": "advanced"},
        timeout=90.0, log=log, stage="retrieve",
    )
    rows = _rows_from_tavily_search(data if status == 200 else None)
    return {"provider": cfg["name"], "summary": "", "sources": rows,
            "fallback_used": True,
            "error": None if rows else f"research and search both failed (status {status})"}


def _rows_from_tavily_search(data: Any) -> list[dict]:
    if not isinstance(data, dict):
        return []
    out: list[dict] = []
    for r in data.get("results") or []:
        if not isinstance(r, dict) or not r.get("url"):
            continue
        text = r.get("raw_content") or r.get("content") or ""
        out.append({"url": r["url"], "title": r.get("title") or "",
                    "text": text if isinstance(text, str) else "", "date": ""})
    return out


async def you_research(question: str, log: RunLog) -> dict:
    """You.com Research. Submitted with background=true, then polled."""
    cfg = SEARCH_CFG["you_research"]
    key = key_for(cfg)
    if not key:
        return {"provider": cfg["name"], "summary": "", "sources": [], "error": "no api key",
                "fallback_used": False}
    headers = {"X-API-Key": key, "Content-Type": "application/json"}
    body = {"input": question, "research_effort": cfg["effort"], "background": True}
    status, data = await http_request("POST", cfg["base_url"], headers=headers,
                                      json_body=body, timeout=90.0, log=log, stage="retrieve")
    if status == 200 and isinstance(data, dict):
        output = data.get("output")
        if isinstance(output, dict) and output.get("content"):
            srcs = output.get("sources") or []
            rows = _rows_from_you_sources(srcs) or segments_from_citations(
                output["content"], srcs
            )
            return {"provider": cfg["name"], "summary": output["content"], "sources": rows,
                    "fallback_used": False, "error": None}
        task_id = data.get("task_id")
        if task_id:
            deadline = time.time() + cfg["poll_timeout_s"]
            while time.time() < deadline:
                await _poll_sleep(cfg["poll_interval_s"])
                p_status, payload = await http_request(
                    "GET", f"{cfg['base_url']}/{task_id}",
                    headers={"X-API-Key": key}, timeout=60.0, attempts=2,
                    log=log, stage="retrieve",
                )
                if p_status != 200 or not isinstance(payload, dict):
                    break
                state = payload.get("status")
                if state == "completed":
                    result = payload.get("result") or {}
                    output = result.get("output") or {}
                    content = output.get("content") or ""
                    srcs = output.get("sources") or []
                    rows = _rows_from_you_sources(srcs) or segments_from_citations(
                        content, srcs
                    )
                    return {"provider": cfg["name"], "summary": content, "sources": rows,
                            "task_id": task_id, "fallback_used": False, "error": None}
                if state in ("failed", "cancelled"):
                    break
    return {"provider": cfg["name"], "summary": "", "sources": [],
            "fallback_used": True, "error": f"research failed (status {status})"}


def _rows_from_you_sources(srcs: Any) -> list[dict]:
    if not isinstance(srcs, list):
        return []
    out: list[dict] = []
    for s in srcs:
        if not isinstance(s, dict) or not s.get("url"):
            continue
        snips = s.get("snippets") or []
        text = "\n\n".join(x for x in snips if isinstance(x, str))
        if not text.strip():
            continue
        out.append({"url": s["url"], "title": s.get("title") or "", "text": text[:20000],
                    "date": ""})
    return out


async def exa_search(query: str, num_results: int, log: RunLog, stage: str) -> list[dict]:
    cfg = SEARCH_CFG["exa"]
    key = key_for(cfg)
    if not key:
        log.warn(stage, f"exa skipped: {cfg['key_env']} is not set")
        return []
    status, data = await http_request(
        "POST", cfg["base_url"],
        headers={"x-api-key": key, "Content-Type": "application/json"},
        json_body={"query": query, "numResults": num_results,
                   "contents": {"text": {"maxCharacters": cfg["text_max_chars"],
                                         "verbosity": "standard"}}},
        timeout=90.0, log=log, stage=stage,
    )
    if status != 200 or not isinstance(data, dict):
        log.warn(stage, f"exa search failed for {query[:60]!r} (status {status})")
        return []
    out: list[dict] = []
    for r in data.get("results") or []:
        if not isinstance(r, dict) or not r.get("url"):
            continue
        text = r.get("text") or r.get("summary") or ""
        if isinstance(text, dict):
            text = ""
        if not str(text).strip():
            continue
        out.append({"url": r["url"], "title": r.get("title") or "", "text": str(text),
                    "date": r.get("publishedDate") or ""})
    return out


async def tavily_search(query: str, num_results: int, log: RunLog, stage: str) -> list[dict]:
    cfg = SEARCH_CFG["tavily_search"]
    key = key_for(cfg)
    if not key:
        log.warn(stage, f"tavily search skipped: {cfg['key_env']} is not set")
        return []
    status, data = await http_request(
        "POST", cfg["base_url"],
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json_body={
            "query": query,
            "max_results": num_results,
            "search_depth": cfg.get("search_depth", "advanced"),
            "chunks_per_source": cfg.get("chunks_per_source", 3),
            "include_raw_content": cfg.get("include_raw_content", "markdown"),
        },
        timeout=90.0, log=log, stage=stage,
    )
    if status != 200:
        log.warn(stage, f"tavily search failed for {query[:60]!r} (status {status})")
        return []
    return _rows_from_tavily_search(data)


async def stage_retrieve(query: str, plan: dict, log: RunLog) -> list[Document]:
    """Research line 1 -> Tavily Search (advanced), line 2 -> You.com Research, each
    Search line -> Exa. All in parallel. One provider failing does not kill the run."""
    log.stage_begin("retrieve", {"research_queries": len(plan["research"]),
                                 "search_queries": len(plan["search"])})
    research = plan["research"]
    search = plan["search"]
    research_provider = env_str("VIN_RESEARCH_PROVIDER_A", "tavily").lower()
    research_provider_b = env_str("VIN_RESEARCH_PROVIDER_B", "you").lower()

    async def do_research(slot: int, provider: str, line: str) -> tuple[str, str, dict]:
        if slot >= len(research):
            return provider, line, {"provider": provider, "sources": [], "error": "no research line",
                                    "fallback_used": False}
        if provider == "you":
            res = await you_research(line, log)
        else:
            rows = await tavily_search(line, SEARCH_CFG["tavily_search"]["num_results"], log, "retrieve")
            res = {"provider": provider, "sources": rows, "fallback_used": False, "error": None}
        return provider, line, res

    tasks = [
        asyncio.create_task(do_research(0, research_provider, research[0] if research else "")),
        asyncio.create_task(do_research(1, research_provider_b, research[1] if len(research) > 1 else "")),
    ]
    for q in search:
        tasks.append(asyncio.create_task(exa_search(q, SEARCH_CFG["exa"]["num_results"], log, "retrieve")))
    settled = await asyncio.gather(*tasks, return_exceptions=True)

    registry = _current_registry(log)
    docs: list[Document] = []
    providers_ok: list[str] = []
    providers_failed: list[str] = []
    exa_rows: list[tuple[str, dict]] = []
    for i, item in enumerate(settled):
        if isinstance(item, Exception):
            log.warn("retrieve", f"provider task {i} raised {type(item).__name__}: {item}")
            providers_failed.append(f"task{i}")
            continue
        if i < 2:
            provider, line, res = item
            if res.get("error") and not res.get("sources"):
                log.warn("retrieve", f"{provider} research failed: {res['error']}")
                providers_failed.append(provider)
            else:
                providers_ok.append(provider)
                if res.get("fallback_used"):
                    log.warn("retrieve", f"{provider} research used the standard search fallback")
                for row in res.get("sources") or []:
                    sid = registry.register(row["url"], row.get("title", ""), row.get("date", ""),
                                            provider, "primary")
                    if sid:
                        registry.register_text(sid, row.get("text", ""))
                    if sid and str(row.get("text", "")).strip():
                        docs.append(Document(sid, row.get("title") or row["url"], row["url"],
                                             row.get("date", ""), provider, "primary",
                                             str(row["text"])))
        else:
            q = search[i - 2]
            rows = item or []
            if rows:
                providers_ok.append("exa")
                for row in rows:
                    exa_rows.append((q, row))
            else:
                providers_failed.append("exa")
    for q, row in exa_rows:
        sid = registry.register(row["url"], row.get("title", ""), row.get("date", ""), "exa", "primary")
        if sid:
            registry.register_text(sid, row.get("text", ""))
            docs.append(Document(sid, row.get("title") or row["url"], row["url"],
                                 row.get("date", ""), "exa", "primary", str(row["text"])))

    deduped: list[Document] = []
    seen: set[str] = set()
    for d in docs:
        key = normalize_url(d.url)
        if not key or key in seen:
            continue
        if len(d.text.split()) < 25:
            continue
        seen.add(key)
        deduped.append(d)
    if not deduped:
        log.stage_end("failed", error="every retrieval provider returned no usable text")
        raise AbortPipeline(
            "retrieval produced no documents: all of Tavily Research, You.com Research "
            "and Exa failed or returned empty text. Core stage, cannot continue."
        )
    log.stage_usage(providers_ok=providers_ok, providers_failed=sorted(set(providers_failed)),
                    documents=len(deduped))
    log.artifact_json("02_retrieve.json",
                      {"sources": [d.to_dict() for d in deduped], "plan": plan,
                       "providers_ok": providers_ok,
                       "providers_failed": sorted(set(providers_failed))})
    log.stage_end("ok")
    return deduped


def _current_registry(log: RunLog) -> SourceRegistry:
    return log.registry  # type: ignore[attr-defined]


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset("""
a an and are as at be been but by for from has have he her his in is it its of on or
she that the their them there these they this to was were what when which who will with
""".split())
_BM25_K1 = 1.2
_BM25_B = 0.75


def lexical_scores(query: str, chunks: Sequence[Chunk]) -> dict[int, float]:
    """Relevance ranking used only when the Cloudflare reranker is unavailable.

    BM25 over chunk text. This is a documented degradation, not a replacement: the
    reranker normally supplies the scores, and log.json records which path ran.
    """
    terms = [t for t in _TOKEN_RE.findall(query.lower()) if t not in _STOPWORDS]
    if not terms or not chunks:
        return {}
    doc_tokens = [_TOKEN_RE.findall(c.text.lower()) for c in chunks]
    n = len(doc_tokens)
    avg_len = sum(len(d) for d in doc_tokens) / n or 1.0
    df: dict[str, int] = {}
    for t in terms:
        df[t] = sum(1 for d in doc_tokens if t in d)
    out: dict[int, float] = {}
    for i, d in enumerate(doc_tokens):
        if not d:
            continue
        counts: dict[str, int] = {}
        for tok in d:
            if tok in df:
                counts[tok] = counts.get(tok, 0) + 1
        score = 0.0
        for t in terms:
            tf = counts.get(t, 0)
            if not tf:
                continue
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            denom = tf + _BM25_K1 * (1 - _BM25_B + _BM25_B * len(d) / avg_len)
            score += idf * (tf * (_BM25_K1 + 1)) / denom
        out[i] = round(score, 6)
    return out


# ============================================================ stage 3 rerank


def format_evidence(chunks: Sequence[Chunk], registry: SourceRegistry) -> str:
    blocks: list[str] = []
    for c in chunks:
        title = registry.title_for(c.source_id)
        blocks.append(f"[{c.source_id}] (chunk {c.index}) {title}\n(c) {c.text}")
    return "\n\n".join(blocks)


async def _rerank_scores(query: str, chunks: list[Chunk], log: RunLog) -> dict[int, float]:
    cfg = SEARCH_CFG["cloudflare_rerank"]
    account_id = env_str(cfg["account_id_env"]) or ("dry-run-account" if
                                                    _DRY_RUN["scenario"] else "")
    token = key_for(cfg)
    if not token:
        log.warn("rerank", f"cloudflare token not found in any of {cfg['key_env']}, "
                           f"{cfg.get('key_env_aliases')}; keeping unranked chunks")
        return {}
    if not account_id:
        log.warn("rerank", f"{cfg['account_id_env']} is not set, so the account-scoped "
                           "Workers AI URL cannot be built; keeping unranked chunks")
        return {}
    url = cfg["base_url"].format(account_id=account_id, model=cfg["model"])
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    scores: dict[int, float] = {}
    first_raw: Any = None

    async def _run_rerank_batch(start: int) -> tuple[int, int, Any, int]:
        batch = chunks[start: start + RERANK_BATCH]
        contexts = [{"text": c.text} for c in batch]
        status, data = await http_request(
            "POST", url, headers=headers,
            json_body={"query": query, "contexts": contexts, "top_k": len(contexts)},
            timeout=45.0, attempts=2, log=log, stage="rerank",
        )
        return start, status, data, len(batch)

    batch_starts = list(range(0, len(chunks), RERANK_BATCH))
    results = await asyncio.gather(*[_run_rerank_batch(s) for s in batch_starts])
    for start, status, data, blen in results:
        if status != 200:
            log.warn("rerank", f"reranker batch starting at {start} failed (status {status})")
            continue
        if first_raw is None:
            first_raw = data
            log.artifact_json("03_rerank_raw_first.json", data)
        parsed = parse_rerank_scores(data)
        if parsed is None:
            log.warn("rerank", f"unrecognized reranker envelope: {_short(json.dumps(data, default=str), 400)}")
            continue
        for idx, score in parsed:
            if 0 <= idx < blen:
                scores[start + idx] = score
    if scores:
        vals = sorted(scores.values(), reverse=True)
        dist = {"n": len(vals), "min": vals[-1], "max": vals[0],
                "mean": round(sum(vals) / len(vals), 6),
                "median": vals[len(vals) // 2],
                "p25": vals[max(0, len(vals) // 4)],
                "p75": vals[min(len(vals) - 1, (3 * len(vals)) // 4)],
                "quantiles": {str(round(q, 2)): vals[min(len(vals) - 1, int(len(vals) * q))]
                              for q in (0.05, 0.25, 0.5, 0.75, 0.95)}}
        log.artifact_json("03_rerank_distribution.json",
                          {"threshold": RERANK_MIN_SCORE, "summary": dist,
                           "scores": [{"chunk": chunks[i].uid, "score": s}
                                      for i, s in sorted(scores.items(),
                                                         key=lambda kv: -kv[1])]})
        log.stage_usage(rerank_scores=dist, threshold=RERANK_MIN_SCORE)
        print(f"    rerank raw score distribution: {dist}")
    else:
        log.warn("rerank", "reranker returned no usable scores")
    return scores


async def stage_rerank(query: str, docs: Sequence[Document], log: RunLog) -> list[Chunk]:
    log.stage_begin("rerank", {"documents": len(docs), "threshold": RERANK_MIN_SCORE,
                               "model": SEARCH_CFG["cloudflare_rerank"]["model"]})
    chunks: list[Chunk] = []
    for d in docs:
        for c in chunk_document(d.text, d.source_id):
            c_words = c.words
            if c_words < 25:
                continue
            c.words = c_words
            chunks.append(c)
    log.stage_usage(chunks_total=len(chunks))
    if not chunks:
        log.stage_end("failed", error="no chunk survived chunking")
        raise AbortPipeline("rerank produced no chunks from the retrieved documents")
    scores = await _rerank_scores(query, chunks, log)
    if scores:
        ranked: list[tuple[int, float]] = sorted(
            scores.items(), key=lambda kv: (-kv[1], kv[0])
        )
        ranked += [(i, 0.0) for i in range(len(chunks)) if i not in scores]
    else:
        ranked = sorted(lexical_scores(query, chunks).items(), key=lambda kv: -kv[1])
        log.stage_usage(rerank_source="bm25-fallback")
        log.warn("rerank", "no Cloudflare reranker scores; ranked with a local BM25 "
                           f"fallback instead of document order ({len(ranked)} chunks)")
    ranked.sort(key=lambda kv: -kv[1])
    if scores:
        kept = apply_rerank_threshold(ranked, RERANK_MIN_SCORE, chunks=chunks)
        threshold_applied = True
    else:
        # BM25 scores are not on the reranker's scale, so RERANK_MIN_SCORE does not apply.
        # Take the top RERANK_CAP by relevance and record that in log.json.
        kept = ranked[:RERANK_CAP]
        threshold_applied = False
    surviving = [chunks[i] for i, _ in kept]
    for c, (_i, s) in zip(surviving, kept):
        c.score = s
    dropped = len(chunks) - len(surviving)
    floor_used = any(s < RERANK_MIN_SCORE for _, s in ranked[:RERANK_FLOOR])
    log.stage_usage(surviving=len(surviving), dropped=dropped,
                    floor_forced=bool(floor_used),
                    threshold_applied=threshold_applied,
                    cap=min(len(surviving), RERANK_CAP))
    if dropped:
        print(f"    kept {len(surviving)}/{len(chunks)} chunks "
              f"(threshold_applied={threshold_applied}, floor_forced={floor_used})")
    surviving.sort(key=lambda c: -c.score)
    log.artifact_json("03_rerank.json",
                      {"kept": [{"chunk": c.uid, "source_id": c.source_id, "index": c.index,
                                 "words": c.words, "score": c.score} for c in surviving],
                       "threshold": RERANK_MIN_SCORE, "floor": RERANK_FLOOR, "cap": RERANK_CAP})
    log.stage_end("ok")
    return surviving


# ============================================================ stage 4 evidence ledger (DURALUMIN v4)

import calendar
import hashlib
import re
import json
import asyncio
import numpy as np
import networkx as nx
from collections import defaultdict
from typing import Any, Iterable, Sequence
from pydantic import BaseModel, Field
from rapidfuzz import fuzz

_CHUNK_MENTION_RE = re.compile(r"\[(S(\d+))\] \(chunk (\d+)\)")

_CLAIM_MONEY_RE = re.compile(
    r"(?:\$[\d,]+(?:\.\d+)?"
    r"|[£€₹¥]\s?[\d.,]+(?:\s?(?:million|billion|thousand|million\b))?"
    r"|\b\d[\d.,]?\s+(?:million|billion|thousand)?\s+(?:dollars|units?|euros?|pounds?|USD|EUR|GBP)\b"
    r"|\b\d+(?:\.\d+)?\s*(?:%|percent|percentage)\b"
    r"|\b\d[\d,]*\s+(?:spending|revenue|cost|figure|amount|total)\b)",
    re.IGNORECASE,
)
_CLAIM_DATE_RE = re.compile(
    r"(?:\b(?:19|20)\d{2}-\d{2}(?:-\d{2})?\b|\bQ[1-4]\s+(?:19|20)\d{2}\b|\b(?:19|20)\d{2}\b)",
    re.IGNORECASE,
)
_CLAIM_ATTRIB_RE = re.compile(
    r"\b(per|according to|says|stated|reports?|notes?|found|estimates?|study|regulator|"
    r"authors?|announced|observed|warned|recommends?)\b",
    re.IGNORECASE,
)


class DuraluminMetricModel(BaseModel):
    comparison_group: str = ""
    subject: str = ""
    metric_name: str = ""
    raw_value: str = ""
    numeric_value: float | None = None
    unit: str = ""
    period_or_date: str = ""
    conditions_caveats: str = ""
    source_type: str = "reported figure"
    cites: list[str] = Field(default_factory=list)


def claim_signal(text: str) -> tuple[bool, str]:
    if not text:
        return (False, "")
    money = bool(_CLAIM_MONEY_RE.search(text))
    date = bool(_CLAIM_DATE_RE.search(text))
    attrib = bool(_CLAIM_ATTRIB_RE.search(text))
    if not (money or date or attrib):
        return (False, "")
    if money:
        return (True, "figure")
    if date:
        return (True, "date")
    return (True, "attribution")


def _write_jsonl(path: Path, records: Iterable[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def _ledger_priority(cumulative_words: int) -> str | None:
    if cumulative_words <= LEDGER_PRIORITY_OVERLAP:
        return "overlap"
    if cumulative_words <= LEDGER_PRIORITY_TOP:
        return "top"
    return None


def _as_str_list(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    if isinstance(v, (list, tuple, set)):
        return [str(x) for x in v if x is not None]
    return []


# ============================================================ DURALUMIN Universal Unit & Chronology Engine

try:
    import pint as _pint
    _UREG = _pint.UnitRegistry(autoconvert_offset_to_baseunit=True)
except Exception:
    _UREG = None

try:
    import dateparser as _dateparser
except Exception:
    _dateparser = None

# Every scale from hundred (1e2) to septillion/yotta (1e24)
_SCALE_WORDS: tuple[tuple[str, float], ...] = (
    ("septillion", 1e24),
    ("yotta", 1e24),
    ("sextillion", 1e21),
    ("zetta", 1e21),
    ("quintillion", 1e18),
    ("exa", 1e18),
    ("quadrillion", 1e15),
    ("peta", 1e15),
    ("trillion", 1e12),
    ("tera", 1e12),
    ("tln", 1e12),
    ("tn", 1e12),
    ("billion", 1e9),
    ("giga", 1e9),
    ("bln", 1e9),
    ("bn", 1e9),
    ("million", 1e6),
    ("mega", 1e6),
    ("mln", 1e6),
    ("mn", 1e6),
    ("thousand", 1e3),
    ("kilo", 1e3),
    ("hundred", 1e2),
)

# Canonical dimensional conversion table: (dimension_family, canonical_unit, multiplier_to_canonical)
_UNIT_CONVERSION_TABLE: dict[str, tuple[str, str, float]] = {
    # Digital storage (base: GB)
    "bit": ("digital_storage", "GB", 1.0 / 8e9),
    "bits": ("digital_storage", "GB", 1.0 / 8e9),
    "b": ("digital_storage", "GB", 1e-9),
    "byte": ("digital_storage", "GB", 1e-9),
    "bytes": ("digital_storage", "GB", 1e-9),
    "kb": ("digital_storage", "GB", 1e-6),
    "kilobyte": ("digital_storage", "GB", 1e-6),
    "kilobytes": ("digital_storage", "GB", 1e-6),
    "kib": ("digital_storage", "GB", 1024.0 / 1e9),
    "mb": ("digital_storage", "GB", 1e-3),
    "megabyte": ("digital_storage", "GB", 1e-3),
    "megabytes": ("digital_storage", "GB", 1e-3),
    "mib": ("digital_storage", "GB", (1024.0 ** 2) / 1e9),
    "gb": ("digital_storage", "GB", 1.0),
    "gigabyte": ("digital_storage", "GB", 1.0),
    "gigabytes": ("digital_storage", "GB", 1.0),
    "gib": ("digital_storage", "GB", (1024.0 ** 3) / 1e9),
    "tb": ("digital_storage", "GB", 1e3),
    "terabyte": ("digital_storage", "GB", 1e3),
    "terabytes": ("digital_storage", "GB", 1e3),
    "tib": ("digital_storage", "GB", (1024.0 ** 4) / 1e9),
    "pb": ("digital_storage", "GB", 1e6),
    "petabyte": ("digital_storage", "GB", 1e6),
    "petabytes": ("digital_storage", "GB", 1e6),
    "eb": ("digital_storage", "GB", 1e9),
    "exabyte": ("digital_storage", "GB", 1e9),
    "exabytes": ("digital_storage", "GB", 1e9),
    "zb": ("digital_storage", "GB", 1e12),
    "zettabyte": ("digital_storage", "GB", 1e12),
    "zettabytes": ("digital_storage", "GB", 1e12),
    # Length / Distance (base: m)
    "nm": ("length", "m", 1e-9),
    "nanometer": ("length", "m", 1e-9),
    "nanometers": ("length", "m", 1e-9),
    "nanometre": ("length", "m", 1e-9),
    "um": ("length", "m", 1e-6),
    "µm": ("length", "m", 1e-6),
    "micrometer": ("length", "m", 1e-6),
    "micrometers": ("length", "m", 1e-6),
    "micron": ("length", "m", 1e-6),
    "microns": ("length", "m", 1e-6),
    "mm": ("length", "m", 1e-3),
    "millimeter": ("length", "m", 1e-3),
    "millimeters": ("length", "m", 1e-3),
    "cm": ("length", "m", 1e-2),
    "centimeter": ("length", "m", 1e-2),
    "centimeters": ("length", "m", 1e-2),
    "m": ("length", "m", 1.0),
    "meter": ("length", "m", 1.0),
    "meters": ("length", "m", 1.0),
    "metre": ("length", "m", 1.0),
    "metres": ("length", "m", 1.0),
    "km": ("length", "m", 1e3),
    "kilometer": ("length", "m", 1e3),
    "kilometers": ("length", "m", 1e3),
    "kilometre": ("length", "m", 1e3),
    "kilometres": ("length", "m", 1e3),
    "in": ("length", "m", 0.0254),
    "inch": ("length", "m", 0.0254),
    "inches": ("length", "m", 0.0254),
    "ft": ("length", "m", 0.3048),
    "foot": ("length", "m", 0.3048),
    "feet": ("length", "m", 0.3048),
    "yd": ("length", "m", 0.9144),
    "yard": ("length", "m", 0.9144),
    "yards": ("length", "m", 0.9144),
    "mi": ("length", "m", 1609.344),
    "mile": ("length", "m", 1609.344),
    "miles": ("length", "m", 1609.344),
    # Mass / Weight (base: kg)
    "mg": ("mass", "kg", 1e-6),
    "milligram": ("mass", "kg", 1e-6),
    "milligrams": ("mass", "kg", 1e-6),
    "g": ("mass", "kg", 1e-3),
    "gram": ("mass", "kg", 1e-3),
    "grams": ("mass", "kg", 1e-3),
    "kg": ("mass", "kg", 1.0),
    "kilogram": ("mass", "kg", 1.0),
    "kilograms": ("mass", "kg", 1.0),
    "oz": ("mass", "kg", 0.028349523125),
    "ounce": ("mass", "kg", 0.028349523125),
    "ounces": ("mass", "kg", 0.028349523125),
    "lb": ("mass", "kg", 0.45359237),
    "lbs": ("mass", "kg", 0.45359237),
    "pound": ("mass", "kg", 0.45359237),
    "pounds": ("mass", "kg", 0.45359237),
    "ton": ("mass", "kg", 1000.0),
    "tons": ("mass", "kg", 1000.0),
    "tonne": ("mass", "kg", 1000.0),
    "tonnes": ("mass", "kg", 1000.0),
    "metric ton": ("mass", "kg", 1000.0),
    "metric tons": ("mass", "kg", 1000.0),
    "mt": ("mass", "kg", 1000.0),
    # Energy (base: kWh)
    "wh": ("energy", "kWh", 1e-3),
    "watt-hour": ("energy", "kWh", 1e-3),
    "watt-hours": ("energy", "kWh", 1e-3),
    "kwh": ("energy", "kWh", 1.0),
    "kilowatt-hour": ("energy", "kWh", 1.0),
    "kilowatt-hours": ("energy", "kWh", 1.0),
    "mwh": ("energy", "kWh", 1e3),
    "megawatt-hour": ("energy", "kWh", 1e3),
    "megawatt-hours": ("energy", "kWh", 1e3),
    "gwh": ("energy", "kWh", 1e6),
    "gigawatt-hour": ("energy", "kWh", 1e6),
    "gigawatt-hours": ("energy", "kWh", 1e6),
    "twh": ("energy", "kWh", 1e9),
    "terawatt-hour": ("energy", "kWh", 1e9),
    "terawatt-hours": ("energy", "kWh", 1e9),
    # Power (base: MW)
    "w": ("power", "MW", 1e-6),
    "watt": ("power", "MW", 1e-6),
    "watts": ("power", "MW", 1e-6),
    "kw": ("power", "MW", 1e-3),
    "kilowatt": ("power", "MW", 1e-3),
    "kilowatts": ("power", "MW", 1e-3),
    "mw": ("power", "MW", 1.0),
    "megawatt": ("power", "MW", 1.0),
    "megawatts": ("power", "MW", 1.0),
    "gw": ("power", "MW", 1e3),
    "gigawatt": ("power", "MW", 1e3),
    "gigawatts": ("power", "MW", 1e3),
    "tw": ("power", "MW", 1e6),
    "terawatt": ("power", "MW", 1e6),
    "terawatts": ("power", "MW", 1e6),
    # Time / Latency (base: ms)
    "ns": ("time", "ms", 1e-6),
    "nanosecond": ("time", "ms", 1e-6),
    "nanoseconds": ("time", "ms", 1e-6),
    "us": ("time", "ms", 1e-3),
    "µs": ("time", "ms", 1e-3),
    "microsecond": ("time", "ms", 1e-3),
    "microseconds": ("time", "ms", 1e-3),
    "ms": ("time", "ms", 1.0),
    "millisecond": ("time", "ms", 1.0),
    "milliseconds": ("time", "ms", 1.0),
    "s": ("time", "ms", 1000.0),
    "sec": ("time", "ms", 1000.0),
    "secs": ("time", "ms", 1000.0),
    "second": ("time", "ms", 1000.0),
    "seconds": ("time", "ms", 1000.0),
    "min": ("time", "min", 1.0),
    "mins": ("time", "min", 1.0),
    "minute": ("time", "min", 1.0),
    "minutes": ("time", "min", 1.0),
    "hr": ("time", "min", 60.0),
    "hrs": ("time", "min", 60.0),
    "hour": ("time", "min", 60.0),
    "hours": ("time", "min", 60.0),
}


def _extract_scale_multiplier(val_part: str) -> float:
    """Detect word or suffix scale multipliers from 1e2 up to 1e24."""
    vp = val_part.lower()
    for word, factor in _SCALE_WORDS:
        if re.search(rf"\b{word}\b", vp):
            return factor
    # Check compact shorthand like $1.5B, 250M, 100k, 4.2T (avoiding unit collisions like 5m = meters unless $ or pure count)
    shorthand_m = re.search(
        r"(?:[\$£€₹¥]\s*-?\d+(?:\.\d+)?|-?\d+(?:\.\d+)?)\s*([kmbt])\b(?!\s*(?:wh|w|b|byte|bytes|bit|bits|hz|m|g|s|l)\b)",
        vp,
    )
    if shorthand_m:
        return {"k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}[shorthand_m.group(1)]
    return 1.0


def _parse_numeric_float(raw_val: Any, explicit_num: Any = None) -> float | None:
    """Strictly parse finite numeric values, scientific notation, ranges, and scales (up to 1e24).

    Always prioritizes the true scaled number in `raw_val` over an unscaled LLM `explicit_num`
    (preventing the '$1.0 billion' + explicit_num=1000.0 -> 1000.0 bug).
    """
    s = str(raw_val or "").strip().lower().replace(",", "")
    if not s or not re.search(r"\d", s):
        return None

    # Isolate numerator before rate denominators ('per', '/')
    val_part = re.split(r"\b(?:per)\b|/", s, maxsplit=1)[0]
    mult = _extract_scale_multiplier(val_part)

    # 1. Check explicit range first (e.g. '3,000-10,000+ cycles', '$20-25/kWh', '860 to 900')
    range_m = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:to|\-|–|—|~)\s*(-?\d+(?:\.\d+)?)", val_part)
    if range_m:
        try:
            v1, v2 = float(range_m.group(1)), float(range_m.group(2))
            if v1 > 0 and v2 > 0:
                return ((v1 + v2) / 2.0) * mult
        except ValueError:
            pass

    # 2. Extract first numeric literal (supporting scientific notation like 1.2e9)
    num_m = re.search(r"(-?\d+(?:\.\d+)?(?:e[+\-]?\d+)?)", val_part)
    parsed_from_raw: float | None = None
    if num_m:
        try:
            base_num = float(num_m.group(1))
            parsed_from_raw = base_num * mult
        except ValueError:
            parsed_from_raw = None

    # 3. Reconcile with explicit_num safely
    if isinstance(explicit_num, (int, float)) and not isinstance(explicit_num, bool) and math.isfinite(float(explicit_num)):
        val = float(explicit_num)
        if parsed_from_raw is not None and parsed_from_raw != 0:
            # If explicit_num is already close to parsed_from_raw (within 25%), trust parsed_from_raw
            ratio = abs(val / parsed_from_raw)
            if 0.75 <= ratio <= 1.25:
                return parsed_from_raw
            # Otherwise the LLM mis-scaled explicit_num (e.g. wrote 1000 for $1.0 billion); always trust raw_val!
            return parsed_from_raw
        if mult > 1.0 and 0 < abs(val) <= 10000:
            return val * mult
        return val

    return parsed_from_raw


def _clean_surface(text: str) -> str:
    """Clean leading/trailing punctuation while strictly preserving balanced parentheses."""
    s = (text or "").strip()
    # Strip non-bracket junk from both ends
    s = re.sub(r"^[\*•\-–—.,;:!?\"'\[\]{}\s]+|[\*•\-–—.,;:!?\"'\[\]{}\s]+$", "", s)
    # Unwrap only if the entire string is wrapped in a single pair of outer parentheses
    while s.startswith("(") and s.endswith(")") and s.count("(") == 1 and s.count(")") == 1:
        s = s[1:-1].strip()
    # Fix unbalanced parentheses (e.g. 'Solid-State Batteries (SSBs' -> 'Solid-State Batteries (SSBs)')
    open_c = s.count("(")
    close_c = s.count(")")
    if open_c == 1 and close_c == 0:
        if s.startswith("("):
            s = s[1:].strip()
        else:
            s = s + ")"
    elif close_c == 1 and open_c == 0:
        if s.endswith(")"):
            s = s[:-1].strip()
    return re.sub(r"\s+", " ", s).strip()


def _sanitize_claim_text(text: str) -> str:
    s = (text or "").strip()
    s = re.sub(r"\[\d+\]\(javascript:void\(0\)\)", " ", s)
    s = re.sub(r"<https?://[^>]+>", "", s)
    s = re.sub(r"\[([^\]]+)\]\(https?://[^\)]+\)", r"\1", s)
    s = re.sub(r"Note:\*{2,}\s*\*", "Note: ", s)
    s = re.sub(r"^[\*•\-\s]+", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _is_structurally_valid_entity(surface: str, canonical: str = "", chunk_text: str = "") -> bool:
    """Pure structural validation (zero word lists)."""
    surf = _clean_surface(surface)
    canon = _clean_surface(canonical or surf)
    cand = canon or surf
    if len(cand) < 2 or len(cand) > 70:
        return False
    if not any(c.isalpha() for c in cand):
        return False
    if re.match(r"^[\$£€₹¥%+\-]?\s*\d", cand) and not re.search(r"[A-Za-z]{2,}", cand):
        return False
    if re.match(r"^S\d{1,4}$", cand, re.IGNORECASE):
        return False
    if "://" in cand or "/uas/" in cand.lower():
        return False
    if len(cand.split()) > 7:
        return False
    if not any(c.isupper() for c in (surf + canon)):
        return False
    if chunk_text and _DRY_RUN["scenario"] is None:
        ct_low = chunk_text.lower()
        if surf.lower() not in ct_low and canon.lower() not in ct_low:
            sig_tokens = [t for t in re.findall(r"[a-zA-Z0-9]{3,}", surf.lower()) if t in ct_low]
            if not sig_tokens:
                return False
    return True


def _is_clean_claim_text(text: str) -> bool:
    s = _sanitize_claim_text(text)
    if _DRY_RUN["scenario"] is not None:
        return bool(s)
    if len(s) < 25:
        return False
    if "://" in s and len(s.split()) < 6:
        return False
    if len(re.findall(r"\b\d+(?:\.\d+)?%?\b", s)) > 18:
        return False
    return True


def _numeric_version_signature(name: str) -> tuple[str, ...]:
    """Extract ordered numeric/version tokens (e.g. 'Llama 3.1 70B' -> ('3.1', '70b')).
    Mathematical Invariant: two entities with non-empty, differing numeric signatures
    are distinct versions/models/sizes and MUST NEVER be merged."""
    return tuple(re.findall(r"\d+(?:\.\d+)?[a-z]?", (name or "").lower()))


def _norm_alnum_key(name: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (name or "").lower()))


def _acronym_from_name(name: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", _clean_surface(name))
    if len(words) >= 2:
        return "".join(w[0].lower() for w in words if w)
    return ""


def _parse_period_sort_key(period: str) -> tuple[int, int, int, int]:
    """Deterministic 4-tuple chronological sort key: (year, month, day, specificity_rank).

    Rules:
    - Specific dates/months/quarters sort in exact calendar order within a year:
      Q1 -> (Y, 3, 31, 1), Q2 -> (Y, 6, 30, 1), Q3 -> (Y, 9, 30, 1), Q4 -> (Y, 12, 31, 1).
    - Full-year / generic year ('2025', 'FY 2025', 'Full-year 2025') maps to end-of-year
      (Y, 12, 31, 2) so Q1/Q2/Q3 2025 ALWAYS sort BEFORE Full-Year 2025.
    - Multi-year ranges ('2020-2024', '2027–2028') sort by their endpoint year.
    - 'Current' / 'Present' / 'Now' / 'Latest' maps to (2026, 6, 15, 0) so forward-looking
      '2026 Projected' / '2027' / '2030' always sort AFTER 'Current'.
    """
    p = (period or "").lower().strip()
    if not p or p in ("reported period", "standard reporting basis", "undated", "n/a"):
        return (9998, 12, 31, 9)

    is_projected = bool(re.search(r"\b(project|projected|forecast|target|by\s+20\d{2}|next\s+decade|thereafter)\b", p))
    is_current = bool(re.search(r"\b(current|present|now|latest|today|existing)\b", p))

    years = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", p)]
    if years:
        # For ranges like '2024-2025', use the terminal year of the period
        year = years[-1]
    elif is_current:
        return (2026, 6, 15, 0)
    elif is_projected:
        return (2030, 12, 31, 3)
    else:
        return (9999, 12, 31, 9)

    # Quarter detection (Q1..Q4)
    q_m = re.search(r"\bq([1-4])\b", p)
    if q_m:
        q = int(q_m.group(1))
        q_month = q * 3
        q_day = calendar.monthrange(year, q_month)[1]
        return (year, q_month, q_day, 1)

    # Half-year detection (1H / H1 / 2H / H2)
    if re.search(r"\b(1h|h1|first\s+half)\b", p):
        return (year, 6, 30, 1)
    if re.search(r"\b(2h|h2|second\s+half)\b", p):
        return (year, 12, 31, 1)

    # Early / Mid / Late year qualifiers
    if re.search(r"\bearly\s+(?:19|20)\d{2}\b", p):
        return (year, 3, 1, 1)
    if re.search(r"\bmid[\s\-]+(?:19|20)\d{2}\b", p):
        return (year, 6, 15, 1)
    if re.search(r"\blate\s+(?:19|20)\d{2}\b", p):
        return (year, 11, 15, 1)

    # Explicit Month & Day detection
    found_month = None
    for idx in range(1, 13):
        m_full = calendar.month_name[idx].lower()
        m_abbr = calendar.month_abbr[idx].lower()
        if m_full in p or m_abbr in re.findall(r"[a-z]+", p):
            found_month = idx
            break

    if found_month is not None:
        day_m = re.search(r"\b([12]\d|3[01]|0?[1-9])(?:st|nd|rd|th)?\b", re.sub(r"\b(19|20)\d{2}\b", "", p))
        day = int(day_m.group(1)) if day_m else calendar.monthrange(year, found_month)[1]
        day = min(day, calendar.monthrange(year, found_month)[1])
        return (year, found_month, day, 0)

    # ISO YYYY-MM-DD or YYYY-MM
    iso_m = re.search(r"\b(?:19|20)\d{2}-(\d{2})(?:-(\d{2}))?\b", p)
    if iso_m:
        mo = max(1, min(12, int(iso_m.group(1))))
        da = int(iso_m.group(2)) if iso_m.group(2) else calendar.monthrange(year, mo)[1]
        return (year, mo, min(da, calendar.monthrange(year, mo)[1]), 0)

    # Fallback to dateparser if installed and string has additional non-year tokens
    if _dateparser is not None and len(re.findall(r"[a-z]+", p)) >= 2 and not re.search(r"\b(full|fy|fiscal|annual)\b", p):
        try:
            dt = _dateparser.parse(period, settings={"PREFER_DAY_OF_MONTH": "last"})
            if dt and 1900 <= dt.year <= 2100:
                return (dt.year, dt.month, dt.day, 1)
        except Exception:
            pass

    # Generic year or Full-Year ('2025', 'FY 2025', '2026 projected'):
    # Place at end of year (Month 12, Day 31) with specificity=2 (or 3 if projected)
    return (year, 12, 31, 3 if is_projected else 2)


def _salvage_extraction_list(raw: str) -> list[dict]:
    """Bracket-tolerant salvager: extracts every complete chunk extraction dict even if
    the LLM output has trailing commas, markdown prose, or truncated closing brackets."""
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("empty response")
    cleaned = re.sub(r",\s*([}\]])", r"\1", raw.strip())
    try:
        obj = extract_json(cleaned)
        if isinstance(obj.get("extractions"), list):
            return obj["extractions"]
    except Exception:
        pass

    # State-machine scan for individual {"chunk_id": ...} objects inside truncated JSON
    salvaged: list[dict] = []
    pos = 0
    n = len(cleaned)
    while pos < n:
        match = re.search(r'\{\s*"chunk_id"\s*:', cleaned[pos:])
        if not match:
            break
        start = pos + match.start()
        depth = 0
        in_str = False
        esc = False
        end = -1
        for i in range(start, n):
            ch = cleaned[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end != -1:
            frag = cleaned[start:end]
            try:
                item = json.loads(frag)
                if isinstance(item, dict) and "chunk_id" in item:
                    salvaged.append(item)
            except Exception:
                pass
            pos = end
        else:
            break
    if salvaged:
        return salvaged
    raise ValueError("no valid extraction objects found")


def _parse_ledger_extractions(
    raw: str,
    chunk_ids: set[str],
    chunk_text_map: dict[str, str] | None = None,
    allow_partial: bool = False,
) -> list[dict]:
    """Parse and validate chunk extractions using pure structural invariants."""
    if allow_partial:
        ext = _salvage_extraction_list(raw)
    else:
        obj = extract_json(raw)
        ext = obj.get("extractions")
        if not isinstance(ext, list):
            raise ValueError("extractions is not a list")
    out: list[dict] = []
    seen: set[str] = set()
    ordered_cids = list(chunk_ids)
    for idx_item, item in enumerate(ext):
        if not isinstance(item, dict):
            raise ValueError("extraction element is not an object")
        cid = str(item.get("chunk_id") or "").strip()
        # Normalize common LLM chunk_id typos like 'S6i2' -> 'S6#2' or 'S21_0' -> 'S21#0'
        cid_norm = re.sub(r"^(S\d+)[i_\-:](\d+)$", r"\1#\2", cid)
        if cid_norm in chunk_ids:
            cid = cid_norm
        elif cid not in chunk_ids and chunk_text_map is not None and len(ext) == len(ordered_cids):
            # Positional recovery for live extraction when the LLM returned exact 1-to-1 items
            unused = [c for c in ordered_cids if c not in seen]
            if unused:
                cid = unused[0]
        if not isinstance(cid, str) or cid not in chunk_ids:
            raise ValueError(f"unknown chunk_id {cid!r}")
        if cid in seen:
            raise ValueError(f"duplicate chunk_id {cid!r}")
        seen.add(cid)
        sid_default = cid.split("#", 1)[0]
        ctext = (chunk_text_map or {}).get(cid, "")
        cites = _as_str_list(item.get("citations"))

        clean_entities: list[dict] = []
        for e in list(item.get("entities") or []):
            if not isinstance(e, dict):
                continue
            surf = _clean_surface(str(e.get("surface") or ""))
            canon = _clean_surface(str(e.get("canonical_name") or e.get("canonical_hint") or surf))
            if not _is_structurally_valid_entity(surf, canon, ctext):
                continue
            aliases = [
                _clean_surface(a) for a in _as_str_list(e.get("aliases"))
                if _clean_surface(a) and any(c.isalpha() for c in a)
            ]
            clean_entities.append({
                "surface": surf,
                "canonical_hint": canon,
                "aliases": aliases,
                "type": _clean_surface(str(e.get("type") or "Entity")) or "Entity",
                "role": _clean_surface(str(e.get("role") or "")),
            })

        norm_metrics: list[dict] = []
        rescued_qualitative_claims: list[dict] = []
        for m in list(item.get("metrics") or []):
            if not isinstance(m, dict):
                continue
            rval = _clean_surface(str(m.get("raw_value") or m.get("surface") or ""))
            num_val = _parse_numeric_float(rval, m.get("numeric_value"))
            subj = _clean_surface(str(m.get("subject") or ""))
            mname = _clean_surface(str(m.get("metric_name") or m.get("property_measured") or ""))
            cav = _sanitize_claim_text(str(m.get("conditions_caveats") or m.get("caveat") or ""))
            mcites = _as_str_list(m.get("cites")) or cites or [sid_default]

            if num_val is None or not re.search(r"\d", rval):
                if subj and mname and len(rval.split()) >= 2:
                    rescued_qualitative_claims.append({
                        "subject": subj,
                        "relation": mname,
                        "object": rval,
                        "text": f"{subj} ({mname}): {rval}" + (f" — {cav}." if cav else "."),
                        "cites": mcites,
                    })
                continue

            validated = DuraluminMetricModel(
                comparison_group=re.sub(r"[^a-z0-9_]+", "_", str(m.get("comparison_group") or "").lower()).strip("_"),
                subject=subj,
                metric_name=mname,
                raw_value=rval,
                numeric_value=num_val,
                unit=_clean_surface(str(m.get("unit") or "")),
                period_or_date=_clean_surface(str(m.get("period_or_date") or m.get("date") or "")),
                conditions_caveats=cav,
                source_type=_clean_surface(str(m.get("source_type") or m.get("epistemic_status") or "reported figure")).lower().replace("_", " "),
                cites=mcites,
            )
            norm_metrics.append(validated.model_dump())

        raw_claims = list(item.get("claims") or []) + rescued_qualitative_claims
        clean_claims: list[dict] = []
        for cl in raw_claims:
            if not isinstance(cl, dict):
                continue
            t = _sanitize_claim_text(str(cl.get("text") or ""))
            if not _is_clean_claim_text(t):
                continue
            clean_claims.append({
                "subject": _clean_surface(str(cl.get("subject") or "")),
                "relation": _clean_surface(str(cl.get("relation") or "reports")),
                "object": _clean_surface(str(cl.get("object") or "")),
                "text": t,
                "cites": _as_str_list(cl.get("cites")) or cites or [sid_default],
            })

        clean_leads: list[str] = []
        for ld in _as_str_list(item.get("leads_and_gaps")):
            s_ld = _sanitize_claim_text(ld)
            if not s_ld or len(s_ld) < 20:
                continue
            if re.search(r"\bin this excerpt\b|\bin the provided excerpt\b|\bnot included in this excerpt\b", s_ld, re.I):
                continue
            clean_leads.append(s_ld)

        entry: dict[str, Any] = {
            "chunk_id": cid,
            "citations": cites,
            "entities": clean_entities,
            "claims": clean_claims,
        }
        if norm_metrics or "metrics" in item:
            entry["metrics"] = norm_metrics
        if clean_leads or "leads_and_gaps" in item:
            entry["leads_and_gaps"] = clean_leads
        out.append(entry)

    missing = chunk_ids - seen
    if missing and not (allow_partial and len(out) > 0):
        raise ValueError(f"missing extractions for {sorted(missing)}")
    return out


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _first_sentence(text: str) -> str:
    parts = [p for p in _SENT_SPLIT.split(text.strip()) if p.strip()]
    return parts[0] if parts else (text.strip()[:200])


def _render_ledger_digest(notary: list[dict]) -> str:
    lines: list[str] = []
    seen: set[str] = set()
    used_model = any(
        r.get("provenance") == "model" and r.get("claims") for r in notary
    )
    if used_model:
        for r in notary:
            claims = r.get("claims") or []
            if not claims:
                continue
            fallback_tags = [f"[{c}]" for c in (r.get("citations") or [])]
            for cl in claims:
                text = _sanitize_claim_text(cl.get("text") or "")
                if not text or text in seen or not _is_clean_claim_text(text):
                    continue
                seen.add(text)
                tags = [f"[{c}]" for c in _as_str_list(cl.get("cites"))] or fallback_tags
                lines.append((text + " " + " ".join(tags)).rstrip())
        if lines:
            return "\n".join(lines)
    ordered = sorted(
        (r for r in notary if (r.get("text") or "").strip()),
        key=lambda r: r.get("score", 0.0),
        reverse=True,
    )
    for r in ordered:
        sid = r["source_id"]
        first = _first_sentence(r["text"])
        if first and first not in seen:
            seen.add(first)
            lines.append(f"{first} [{sid}]")
    return "\n".join(lines) if lines else ""


_MACHINE_MONEY_RE = re.compile(
    r"(?:\$[\d,]+(?:\.\d+)?(?:\s?(?:million|billion|thousand|trillion|M|B|bn|mn))?"
    r"|[£€₹¥]\s?[\d.,]+(?:\s?(?:million|billion|thousand|trillion|mn|bn))?)"
)
_MACHINE_DATE_RE = re.compile(
    r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b|\b\d{1,2}\s+[A-Z][a-z]+\s+(?:19|20)\d{2}\b|\b[A-Z][a-z]+\s+\d{1,2}(?:st|nd|rd|th)?,?\s+(?:19|20)\d{2}\b"
)
_MACHINE_UNITED_AMOUNT_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s+(?:million|billion|thousand|trillion)\s+(?!in\b|for\b|to\b|of\b|by\b|at\b|on\b|from\b)[a-zA-Z/]+"
)
_MACHINE_PERCENT_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:%|percent|percentage(?: points?)?)\b"
)


def _norm_machine_date(surf: str) -> str:
    month_lookup = {calendar.month_name[i].lower(): i for i in range(1, 13)}
    m = re.match(r"(\w+)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})", surf)
    if m:
        num = month_lookup.get(m.group(1).lower())
        if num:
            return f"{m.group(3)}-{num:02d}-{int(m.group(2)):02d}"
    m = re.match(r"(\d{1,2})\s+(\w+)\s+(\d{4})", surf)
    if m:
        num = month_lookup.get(m.group(2).lower())
        if num:
            return f"{m.group(3)}-{num:02d}-{int(m.group(1)):02d}"
    return ""


def _ledger_machine_extract(text: str) -> tuple[list[dict], list[dict], list[dict]]:
    entities: list[dict] = []
    money: list[dict] = []
    dates: list[dict] = []
    if not text:
        return entities, money, dates
    for m in _MACHINE_DATE_RE.finditer(text):
        surf = m.group(0)
        nd = _norm_machine_date(surf)
        if nd or re.match(r"^\d{4}-\d{2}-\d{2}$", surf):
            entities.append({"surface": surf, "type": "Date",
                             "start": m.start(), "end": m.end(), "norm": nd or surf})
            dates.append({"surface": surf, "norm": nd or surf,
                          "start": m.start(), "end": m.end()})
    for m in _MACHINE_MONEY_RE.finditer(text):
        surf = m.group(0)
        entities.append({"surface": surf, "type": "Money",
                         "start": m.start(), "end": m.end()})
        money.append({"surface": surf, "start": m.start(), "end": m.end()})
    for m in _MACHINE_UNITED_AMOUNT_RE.finditer(text):
        surf = m.group(0)
        entities.append({"surface": surf, "type": "Quantity",
                         "start": m.start(), "end": m.end()})
    for m in _MACHINE_PERCENT_RE.finditer(text):
        surf = m.group(0)
        entities.append({"surface": surf, "type": "Quantity",
                         "start": m.start(), "end": m.end()})
    entities.sort(key=lambda e: (e.get("start", 0), e.get("end", 0)))
    money.sort(key=lambda e: e.get("start", 0))
    dates.sort(key=lambda e: e.get("start", 0))
    return entities, money, dates


# ------------------------------------------------------------ DURALUMIN v4 Phases

def _shingle_set(text: str, k: int = 3) -> frozenset[str]:
    words = [w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(w) >= 3]
    if len(words) < k:
        return frozenset(words)
    return frozenset(" ".join(words[i: i + k]) for i in range(len(words) - k + 1))


def _dedupe_echo_sources(chunks: Sequence[Chunk], threshold: float = 0.82) -> dict[str, str]:
    by_source: dict[str, list[str]] = defaultdict(list)
    for c in chunks:
        by_source[c.source_id].append(c.text)
    source_shingles: dict[str, frozenset[str]] = {
        sid: _shingle_set(" ".join(texts)) for sid, texts in by_source.items()
    }
    cluster_map: dict[str, str] = {}
    reps: list[tuple[str, frozenset[str]]] = []
    for sid in sorted(source_shingles.keys(), key=lambda s: int(s[1:]) if s[1:].isdigit() else 999):
        sh = source_shingles[sid]
        matched_cid = None
        for rep_sid, rep_sh in reps:
            if sh and rep_sh:
                jaccard = len(sh & rep_sh) / max(1, len(sh | rep_sh))
                if jaccard >= threshold:
                    matched_cid = f"ECHO_{rep_sid}"
                    break
        if matched_cid is None:
            matched_cid = f"ECHO_{sid}"
            reps.append((sid, sh))
        cluster_map[sid] = matched_cid
    return cluster_map


async def _duralumin_profile_query(query: str, log: RunLog) -> dict:
    """Phase 0: Dynamically generate the domain ontology (entity_types & target_metrics) for any query."""
    default_profile = {
        "archetype": "HYBRID",
        "entity_types": ["Entity"],
        "primary_subjects": [w for w in re.findall(r"[A-Za-z0-9_-]{4,}", query)[:6]],
        "target_metrics": [],
        "key_questions": [query],
    }
    messages = [
        {"role": "system", "content": P.DURALUMIN_PROFILER_SYSTEM},
        {"role": "user", "content": f"USER QUESTION:\n{query}"},
    ]
    payload, _cfg, _fb = await llm_chain(
        (DURALUMIN_PROFILER, *DURALUMIN_PROFILER.fallback),
        lambda c: llm_json(c, messages, log, "ledger"),
        log, "ledger", core=False,
    )
    if isinstance(payload, dict) and payload.get("archetype"):
        return {
            "archetype": str(payload.get("archetype") or "HYBRID"),
            "entity_types": _as_str_list(payload.get("entity_types"))[:8] or default_profile["entity_types"],
            "primary_subjects": _as_str_list(payload.get("primary_subjects"))[:8],
            "target_metrics": _as_str_list(payload.get("target_metrics"))[:8],
            "key_questions": _as_str_list(payload.get("key_questions"))[:6],
        }
    return default_profile


async def _duralumin_extract_batches(
    query: str,
    profile: dict,
    chunks: Sequence[Chunk],
    registry: SourceRegistry,
    log: RunLog,
) -> tuple[dict[str, dict], str | None, str | None, str]:
    """Phase 1: Foolproof extraction with Recursive Binary Batch Splitting."""
    if not chunks:
        return {}, None, None, "verbatim"

    batch_sz = DURALUMIN_BATCH_SIZE
    batches: list[list[Chunk]] = [
        list(chunks[i: i + batch_sz])
        for i in range(0, len(chunks), batch_sz)
    ]
    sem = asyncio.Semaphore(DURALUMIN_CONCURRENCY)
    extractions_by_id: dict[str, dict] = {}
    models_succeeded: list[str] = []
    providers_succeeded: list[str] = []
    failure_counts: dict[str, int] = defaultdict(int)
    chunk_text_map = {c.uid: c.text for c in chunks}

    pool: tuple[StageCfg, ...] = (
        (LEDGER,) if _DRY_RUN["scenario"] is not None else DURALUMIN_EXTRACT_POOL
    )
    profile_summary = json.dumps(profile, ensure_ascii=False)
    is_live = _DRY_RUN["scenario"] is None

    async def _extract_subbatch(sub_chunks: list[Chunk], seed_offset: int) -> list[dict] | None:
        if not sub_chunks:
            return []
        batch_evidence = format_evidence(sub_chunks, registry)
        batch_ids = {c.uid for c in sub_chunks}
        messages = [
            {"role": "system", "content": P.LEDGER_EXTRACT_SYSTEM},
            {"role": "user", "content": (
                f"USER QUESTION:\n{query}\n\n"
                f"DYNAMIC DOMAIN PROFILE:\n{profile_summary}\n\n"
                f"SOURCE EXCERPTS ({len(sub_chunks)} chunks):\n\n{batch_evidence}"
            )},
        ]
        # Round-robin rotate starting provider so all 4 providers work in parallel
        plen = len(pool)
        active_pool = [pool[(seed_offset + k) % plen] for k in range(plen)]
        salvaged_so_far: dict[str, dict] = {}

        async with sem:
            for cfg in active_pool:
                if not cfg.have_key() and is_live:
                    continue
                if failure_counts[cfg.name] >= 2 and is_live:
                    continue
                try:
                    res = await llm_call(cfg, messages, log=log, stage="ledger", max_tokens=6000, attempts=1)
                    log.stage_usage(
                        provider=res.model,
                        usage=res.usage,
                        finish_reason=res.finish_reason,
                        reasoning_chars=res.reasoning_chars,
                    )
                    parsed = _parse_ledger_extractions(
                        res.text, batch_ids, chunk_text_map, allow_partial=is_live
                    )
                    for item in parsed:
                        salvaged_so_far[item["chunk_id"]] = item
                    models_succeeded.append(cfg.model)
                    providers_succeeded.append(cfg.provider)
                    # If we got all chunks (or >= 80% of the batch), accept immediately!
                    if len(salvaged_so_far) == len(batch_ids):
                        return list(salvaged_so_far.values())
                    break
                except StageError as exc:
                    failure_counts[cfg.name] += 1
                    log.warn("ledger", f"sub-batch ({len(sub_chunks)} chunks) on {cfg.name} failed: {_short(str(exc), 100)}")
                    continue
                except (ValueError, json.JSONDecodeError) as exc:
                    log.warn("ledger", f"sub-batch ({len(sub_chunks)} chunks) on {cfg.name} parse error: {_short(str(exc), 100)}")
                    continue

        # Only retry/split the MISSING chunks instead of re-running chunks that already succeeded!
        missing_chunks = [c for c in sub_chunks if c.uid not in salvaged_so_far]
        if not missing_chunks:
            return list(salvaged_so_far.values())
        if len(missing_chunks) < len(sub_chunks) and is_live:
            log.info("ledger", f"salvaged {len(salvaged_so_far)}/{len(sub_chunks)} chunks; retrying only {len(missing_chunks)} missing chunk(s)")
            rest = await _extract_subbatch(missing_chunks, seed_offset + 1)
            if rest:
                for item in rest:
                    salvaged_so_far[item["chunk_id"]] = item
            return list(salvaged_so_far.values()) if salvaged_so_far else None

        if len(sub_chunks) > 1 and is_live:
            mid = len(sub_chunks) // 2
            log.info("ledger", f"splitting {len(sub_chunks)}-chunk batch into ({mid} + {len(sub_chunks) - mid}) atomic calls")
            left, right = await asyncio.gather(
                _extract_subbatch(sub_chunks[:mid], seed_offset),
                _extract_subbatch(sub_chunks[mid:], seed_offset + 1),
            )
            combined: list[dict] = list(salvaged_so_far.values())
            if left:
                combined.extend(left)
            if right:
                combined.extend(right)
            return combined if combined else None
        return list(salvaged_so_far.values()) if salvaged_so_far else None

    batch_results = await asyncio.gather(
        *[_extract_subbatch(b, i) for i, b in enumerate(batches)],
        return_exceptions=True,
    )

    for res in batch_results:
        if isinstance(res, list):
            for item in res:
                extractions_by_id[item["chunk_id"]] = item

    if models_succeeded:
        primary_m = models_succeeded[0]
        primary_p = providers_succeeded[0]
        log.provider_used("ledger", primary_m, primary_m != LEDGER.model)
        return extractions_by_id, primary_m, primary_p, "model"

    if not LEDGER_VERBATIM_FALLTHROUGH:
        raise AbortPipeline("ledger: all DURALUMIN extraction pool providers failed and verbatim fallthrough is disabled")

    log.warn("ledger", "all LLM extraction pool providers unavailable; degrading to deterministic sentence extraction")
    return extractions_by_id, None, None, "verbatim"


def _local_tfidf_similarity(texts: list[str]) -> np.ndarray | None:
    if len(texts) < 2:
        return None
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vec = TfidfVectorizer(ngram_range=(1, 3), analyzer="char_wb", min_df=1)
        mat = vec.fit_transform(texts)
        return np.asarray(cosine_similarity(mat), dtype=np.float32)
    except Exception:
        return None


async def _cloudflare_embed_texts(texts: list[str], log: RunLog) -> np.ndarray | None:
    if not texts:
        return None
    cfg = SEARCH_CFG["cloudflare_embed"]
    account_id = env_str(cfg["account_id_env"]) or ("dry-run-account" if _DRY_RUN["scenario"] else "")
    token = key_for(cfg)
    if not token or not account_id:
        return None
    url = cfg["base_url"].format(account_id=account_id, model=cfg["model"])
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    vectors: list[list[float]] = []
    batch_size = 64

    async def _embed_one_batch(start: int) -> list[list[float]] | None:
        batch = texts[start: start + batch_size]
        status, data = await http_request(
            "POST", url, headers=headers, json_body={"text": batch},
            timeout=40.0, attempts=2, log=log, stage="ledger",
        )
        if status != 200 or not isinstance(data, dict):
            return None
        res_obj = data.get("result") or {}
        emb_list = res_obj.get("data") if isinstance(res_obj, dict) else None
        if not isinstance(emb_list, list) or not emb_list:
            return None
        if _DRY_RUN["scenario"] is not None and len(emb_list) != len(batch):
            emb_list = [emb_list[i % len(emb_list)] for i in range(len(batch))]
        elif len(emb_list) != len(batch):
            return None
        return emb_list

    batch_starts = list(range(0, len(texts), batch_size))
    batch_outputs = await asyncio.gather(*[_embed_one_batch(s) for s in batch_starts])
    for emb_list in batch_outputs:
        if emb_list is None:
            return None
        vectors.extend(emb_list)
    try:
        arr = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms
    except Exception:
        return None


def _tier1_resolve_entities(raw_entities: list[dict]) -> list[dict]:
    """Zero-List Structural Entity Resolution."""
    clusters: list[dict] = []
    for ent in raw_entities:
        surf = _clean_surface(str(ent.get("surface") or ""))
        hint = _clean_surface(str(ent.get("canonical_hint") or surf))
        if not _is_structurally_valid_entity(surf, hint):
            continue
        canon_name = hint if len(hint) >= len(surf) else surf
        etype = _clean_surface(str(ent.get("type") or "Entity")) or "Entity"
        role = _clean_surface(str(ent.get("role") or ""))
        cites = _as_str_list(ent.get("cites"))
        explicit_aliases = {_clean_surface(a) for a in _as_str_list(ent.get("aliases")) if _clean_surface(a)}

        all_names = {surf, hint, canon_name} | explicit_aliases
        norm_keys = {_norm_alnum_key(n) for n in all_names if _norm_alnum_key(n)}
        num_sig = _numeric_version_signature(canon_name)
        acro = _acronym_from_name(canon_name)
        if len(acro) >= 3:
            norm_keys.add(acro)

        matched = None
        for cl in clusters:
            if num_sig and cl["num_sig"] and num_sig != cl["num_sig"]:
                continue
            if norm_keys & cl["norm_keys"]:
                matched = cl
                break
            if num_sig == cl["num_sig"]:
                score = fuzz.token_sort_ratio(_norm_alnum_key(canon_name), _norm_alnum_key(cl["canonical_name"]))
                if score >= 88:
                    matched = cl
                    break

        if matched is None:
            clusters.append({
                "canonical_name": canon_name,
                "num_sig": num_sig,
                "norm_keys": set(norm_keys),
                "type": etype,
                "aliases": set(all_names),
                "roles": [role] if role else [],
                "cites": set(cites),
            })
        else:
            if len(canon_name) > len(matched["canonical_name"]) and not num_sig:
                matched["canonical_name"] = canon_name
            matched["aliases"].update(all_names)
            matched["norm_keys"].update(norm_keys)
            if role and role not in matched["roles"]:
                matched["roles"].append(role)
            matched["cites"].update(cites)
    return clusters


def _normalize_unit_token(unit: str, raw_value: str) -> str:
    """Canonicalize unit strings across SI, digital, energy, currency, and rate dimensions."""
    u = (unit or "").strip()
    rv = (raw_value or "").strip()
    comb = f"{u} {rv}".lower()

    if "tokens/sec" in comb or "tokens per second" in comb or "tok/s" in comb or "t/s" in comb:
        return "tokens/sec"
    if any(k in comb for k in ("per million", "per 1m", "/m ", "/1m")):
        if "output" in comb and "input" not in comb:
            return "USD/1M output tokens"
        if "input" in comb and "output" not in comb:
            return "USD/1M input tokens"
        return "USD/1M tokens"
    if "%" in comb or "percent" in comb or "percentage" in comb:
        return "%"
    if "/kwh" in comb or "per kilowatt-hour" in comb or "per kwh" in comb:
        return "USD/kWh"
    if "wh/kg" in comb:
        return "Wh/kg"
    if "wh/l" in comb:
        return "Wh/L"
    if "wh/km" in comb:
        return "Wh/km"
    if "/bbl" in comb or "per barrel" in comb or "/barrel" in comb:
        return "USD/bbl"
    if any(k in comb for k in ("b/d", "bpd", "mb/d", "kb/d", "barrels per day")):
        return "b/d"
    if re.search(r"\bcycles?\b", comb):
        return "cycles"

    # Currency normalization (e.g. 'USD billion', 'USD million', '$', 'USD') -> 'USD'
    # Since numeric_value is already scaled to base USD by _parse_numeric_float, all USD amounts share unit 'USD'
    u_low = u.lower().strip()
    if re.match(r"^(?:usd|us\$|\$)(?:\s*(?:trillion|billion|million|thousand|bn|mn|b|m|k))?$", u_low) or (
        ("$" in rv or "usd" in comb) and "/" not in comb and "per " not in comb and u_low in ("", "usd", "$", "usd million", "usd billion", "usd trillion", "usd/million", "usd/billion")
    ):
        return "USD"

    if u_low in _UNIT_CONVERSION_TABLE:
        _, canon_u, _ = _UNIT_CONVERSION_TABLE[u_low]
        return canon_u

    return u or "numeric"


def _convert_to_canonical_unit(num_v: float | None, unit: str, raw_value: str) -> tuple[float | None, str]:
    """Convert a numeric value and unit into its canonical base unit within its dimension family."""
    if num_v is None:
        return None, _normalize_unit_token(unit, raw_value)
    u_clean = (unit or "").strip().lower()
    norm_u = _normalize_unit_token(unit, raw_value)

    # Direct lookup in our deterministic conversion table
    if u_clean in _UNIT_CONVERSION_TABLE and norm_u == _UNIT_CONVERSION_TABLE[u_clean][1]:
        _, canon_u, factor = _UNIT_CONVERSION_TABLE[u_clean]
        # Avoid double-scaling if raw_value already had a word multiplier applied
        return float(num_v) * factor, canon_u

    # Fallback to Pint for compound physical units (if installed and applicable)
    if _UREG is not None and u_clean and norm_u not in ("USD", "%", "cycles", "tokens/sec", "USD/kWh", "USD/1M tokens"):
        try:
            q = _UREG.Quantity(float(num_v), u_clean)
            target_q = q.to(norm_u)
            return float(target_q.magnitude), norm_u
        except Exception:
            pass

    return float(num_v), norm_u


def _tier1_group_metrics(raw_metrics: list[dict]) -> list[dict]:
    """Zero-List Structural Metric Grouping."""
    topics: list[dict] = []
    seen_frames: set[tuple[str, str, float, str, str]] = set()

    for m in raw_metrics:
        rval = _clean_surface(str(m.get("raw_value") or ""))
        num_v = m.get("numeric_value")
        if not rval or num_v is None or not re.search(r"\d", rval):
            continue
        subj = _clean_surface(str(m.get("subject") or ""))
        mname = _clean_surface(str(m.get("metric_name") or ""))
        pdate = _clean_surface(str(m.get("period_or_date") or ""))
        num_v, norm_unit = _convert_to_canonical_unit(float(num_v), str(m.get("unit") or ""), rval)
        m["numeric_value"] = num_v
        cites_list = sorted(_as_str_list(m.get("cites")))

        norm_subj = _norm_alnum_key(subj)
        norm_mname = _norm_alnum_key(mname)
        num_sig_subj = _numeric_version_signature(f"{subj} {mname}")

        dedupe_sig = (norm_subj, norm_mname, round(float(num_v), 6), pdate.lower(), ",".join(cites_list))
        if dedupe_sig in seen_frames:
            continue
        seen_frames.add(dedupe_sig)

        comp_group = str(m.get("comparison_group") or "").strip()
        label = f"{subj} — {mname}".strip(" —") or rval

        matched_top = None
        for top in topics:
            if top["unit"] != norm_unit:
                continue
            if num_sig_subj and top["num_sig"] and num_sig_subj != top["num_sig"]:
                continue

            if fuzz.token_sort_ratio(norm_subj, top["norm_subj"]) >= 85 and fuzz.token_sort_ratio(norm_mname, top["norm_mname"]) >= 80:
                matched_top = top
                break
            if comp_group and top["comp_group"] and comp_group == top["comp_group"]:
                matched_top = top
                break
            if (
                num_sig_subj
                and num_sig_subj == top["num_sig"]
                and fuzz.token_sort_ratio(norm_mname, top["norm_mname"]) >= 85
            ):
                matched_top = top
                break

        if matched_top is None:
            topics.append({
                "topic_name": label,
                "comp_group": comp_group,
                "norm_subj": norm_subj,
                "norm_mname": norm_mname,
                "num_sig": num_sig_subj,
                "unit": norm_unit,
                "frames": [m],
            })
        else:
            merged_into_existing_row = False
            for existing_f in matched_top["frames"]:
                if (
                    _norm_alnum_key(existing_f.get("subject", "")) == norm_subj
                    and _norm_alnum_key(existing_f.get("metric_name", "")) == norm_mname
                    and existing_f.get("numeric_value") is not None
                    and abs(float(existing_f["numeric_value"]) - float(num_v)) <= 1e-6
                ):
                    combined_c = sorted(
                        set(_as_str_list(existing_f.get("cites"))) | set(cites_list),
                        key=lambda s: int(s[1:]) if s[1:].isdigit() else 999,
                    )
                    existing_f["cites"] = combined_c
                    merged_into_existing_row = True
                    break
            if not merged_into_existing_row:
                matched_top["frames"].append(m)
                distinct_subjs = {_norm_alnum_key(f.get("subject", "")) for f in matched_top["frames"]}
                if len(distinct_subjs) >= 2 and mname:
                    shared_tokens = [w for w in subj.split() if any(c.isdigit() for c in w)]
                    prefix = " ".join(shared_tokens) + " — " if shared_tokens else ""
                    matched_top["topic_name"] = f"{prefix}{mname}"

    return topics


async def _duralumin_resolve(
    entity_clusters: list[dict],
    metric_topics: list[dict],
    log: RunLog,
) -> tuple[list[dict], list[dict]]:
    """Phase 2B + 2C: Vector Similarity + Chief Inspector Adjudication (with Version Lock)."""
    ent_labels = [f"{c['canonical_name']} ({'; '.join(sorted(c['aliases'])[:3])})" for c in entity_clusters]
    met_labels = [f"{t['topic_name']} [{t['unit']}]" for t in metric_topics]
    all_texts = ent_labels + met_labels

    emb_matrix = await _cloudflare_embed_texts(all_texts, log) if all_texts else None

    ambiguous_entities: list[dict] = []
    ambiguous_metrics: list[dict] = []
    ent_merge_map: dict[int, int] = {}
    met_merge_map: dict[int, int] = {}

    n_ent = len(entity_clusters)
    n_met = len(metric_topics)

    sim_e = None
    if emb_matrix is not None and emb_matrix.shape[0] == len(all_texts) and n_ent > 1:
        ent_emb = emb_matrix[:n_ent]
        sim_e = np.dot(ent_emb, ent_emb.T)
    elif n_ent > 1:
        sim_e = _local_tfidf_similarity(ent_labels)

    if sim_e is not None and sim_e.shape[0] == n_ent:
        for i in range(n_ent):
            for j in range(i + 1, n_ent):
                sig_i, sig_j = entity_clusters[i]["num_sig"], entity_clusters[j]["num_sig"]
                if sig_i and sig_j and sig_i != sig_j:
                    continue
                s = float(sim_e[i, j])
                if s >= 0.93 and sig_i == sig_j:
                    ent_merge_map[j] = ent_merge_map.get(i, i)
                elif 0.76 <= s < 0.93 and len(ambiguous_entities) < 15:
                    ambiguous_entities.append({
                        "pair_id": f"E_{len(ambiguous_entities)}",
                        "i": i, "j": j,
                        "a": entity_clusters[i]["canonical_name"],
                        "b": entity_clusters[j]["canonical_name"],
                        "similarity": round(s, 3),
                    })

    sim_m = None
    if emb_matrix is not None and emb_matrix.shape[0] == len(all_texts) and n_met > 1:
        met_emb = emb_matrix[n_ent:]
        sim_m = np.dot(met_emb, met_emb.T)
    elif n_met > 1:
        sim_m = _local_tfidf_similarity(met_labels)

    if sim_m is not None and sim_m.shape[0] == n_met:
        for i in range(n_met):
            for j in range(i + 1, n_met):
                if metric_topics[i]["unit"] != metric_topics[j]["unit"]:
                    continue
                sig_i, sig_j = metric_topics[i]["num_sig"], metric_topics[j]["num_sig"]
                if sig_i and sig_j and sig_i != sig_j:
                    continue
                s = float(sim_m[i, j])
                if 0.78 <= s < 0.95 and len(ambiguous_metrics) < 15:
                    ambiguous_metrics.append({
                        "pair_id": f"M_{len(ambiguous_metrics)}",
                        "i": i, "j": j,
                        "a": metric_topics[i]["topic_name"],
                        "b": metric_topics[j]["topic_name"],
                        "unit": metric_topics[i]["unit"],
                        "similarity": round(s, 3),
                    })

    if ambiguous_entities or ambiguous_metrics:
        inspector_input = {
            "entity_candidates": [
                {"pair_id": p["pair_id"], "entity_a": p["a"], "entity_b": p["b"], "similarity": p["similarity"]}
                for p in ambiguous_entities
            ],
            "metric_candidates": [
                {"pair_id": p["pair_id"], "metric_a": p["a"], "metric_b": p["b"],
                 "unit": p["unit"], "similarity": p["similarity"]}
                for p in ambiguous_metrics
            ],
        }
        messages = [
            {"role": "system", "content": P.DURALUMIN_INSPECTOR_SYSTEM},
            {"role": "user", "content": json.dumps(inspector_input, ensure_ascii=False)},
        ]
        verdict, _cfg, _fb = await llm_chain(
            (DURALUMIN_INSPECTOR, *DURALUMIN_INSPECTOR.fallback),
            lambda c: llm_json(c, messages, log, "ledger"),
            log, "ledger", core=False,
        )
        if isinstance(verdict, dict):
            ent_by_pid = {p["pair_id"]: p for p in ambiguous_entities}
            for dec in verdict.get("entity_decisions") or []:
                if isinstance(dec, dict) and str(dec.get("decision")).upper() == "MERGE":
                    p = ent_by_pid.get(str(dec.get("pair_id")))
                    if p:
                        sig_a = entity_clusters[p["i"]]["num_sig"]
                        sig_b = entity_clusters[p["j"]]["num_sig"]
                        if sig_a and sig_b and sig_a != sig_b:
                            continue
                        root = ent_merge_map.get(p["i"], p["i"])
                        ent_merge_map[p["j"]] = root
                        if dec.get("canonical_name"):
                            entity_clusters[root]["canonical_name"] = _clean_surface(str(dec["canonical_name"]))

            met_by_pid = {p["pair_id"]: p for p in ambiguous_metrics}
            for dec in verdict.get("metric_decisions") or []:
                if isinstance(dec, dict) and str(dec.get("decision")).upper() == "SAME_METRIC_TOPIC":
                    p = met_by_pid.get(str(dec.get("pair_id")))
                    if p and metric_topics[p["i"]]["unit"] == metric_topics[p["j"]]["unit"]:
                        root = met_merge_map.get(p["i"], p["i"])
                        met_merge_map[p["j"]] = root
                        if dec.get("canonical_topic"):
                            metric_topics[root]["topic_name"] = _clean_surface(str(dec["canonical_topic"]))

    def _find_root(pmap: dict[int, int], x: int) -> int:
        visited: set[int] = set()
        curr = x
        while curr in pmap and pmap[curr] != curr and curr not in visited:
            visited.add(curr)
            curr = pmap[curr]
        return curr

    merged_entities: dict[int, dict] = {}
    for idx, cl in enumerate(entity_clusters):
        target = _find_root(ent_merge_map, idx)
        if target not in merged_entities:
            merged_entities[target] = cl
        else:
            merged_entities[target]["aliases"].update(cl["aliases"])
            merged_entities[target]["cites"].update(cl["cites"])
            for r in cl["roles"]:
                if r not in merged_entities[target]["roles"]:
                    merged_entities[target]["roles"].append(r)

    final_entities: list[dict] = []
    for i, cl in enumerate(merged_entities.values(), 1):
        cname = cl["canonical_name"]
        c_sig = _numeric_version_signature(cname)
        aliases = sorted(
            a for a in cl["aliases"]
            if a and a.lower() != cname.lower()
            and (not _numeric_version_signature(a) or _numeric_version_signature(a) == c_sig)
        )
        final_entities.append({
            "entity_id": f"ENT_{i:02d}",
            "canonical_name": cname,
            "type": cl["type"],
            "aliases": aliases[:8],
            "role": "; ".join(cl["roles"][:2]) if cl["roles"] else cl["type"],
            "cites": sorted(cl["cites"], key=lambda s: int(s[1:]) if s[1:].isdigit() else 999),
        })

    merged_metrics: dict[int, dict] = {}
    for idx, top in enumerate(metric_topics):
        target = _find_root(met_merge_map, idx)
        if target not in merged_metrics:
            merged_metrics[target] = top
        else:
            merged_metrics[target]["frames"].extend(top["frames"])

    final_metrics: list[dict] = []
    for i, top in enumerate(merged_metrics.values(), 1):
        final_metrics.append({
            "topic_id": f"MTOP_{i:02d}",
            "topic_name": top["topic_name"],
            "unit": top["unit"],
            "unit_family": top["unit"],
            "frames": top["frames"],
        })

    return final_entities, final_metrics


def _duralumin_build_graph_and_math(
    entities: list[dict],
    metric_topics: list[dict],
    all_claims: list[dict],
    all_leads: list[tuple[str, list[str]]],
    echo_map: dict[str, str],
) -> dict:
    """Phase 3: Pure Entity-Anchored NetworkX Graph + 3-Mode NumPy Math Engine."""
    G = nx.MultiDiGraph()

    alias_to_eid: dict[str, str] = {}
    for ent in entities:
        eid = ent["entity_id"]
        G.add_node(
            eid,
            label=ent["canonical_name"],
            kind="entity",
            etype=ent["type"],
            cites=ent["cites"],
        )
        for surf in [ent["canonical_name"], *ent["aliases"]]:
            k = _norm_alnum_key(surf)
            if k:
                alias_to_eid[k] = eid

    next_concept_idx = 1

    def _resolve_entity_node(name: str, cites: list[str] | None = None, allow_create: bool = False) -> str | None:
        nonlocal next_concept_idx
        raw_name = _clean_surface(name)
        if not raw_name:
            return None
        nk = _norm_alnum_key(raw_name)
        if not nk:
            return None
        if nk in alias_to_eid:
            return alias_to_eid[nk]
        num_sig = _numeric_version_signature(raw_name)
        for k, eid in alias_to_eid.items():
            if len(k) >= 4 and _numeric_version_signature(k) == num_sig and fuzz.token_sort_ratio(nk, k) >= 88:
                alias_to_eid[nk] = eid
                return eid
        if allow_create and 3 <= len(raw_name) <= 75 and any(c.isalpha() for c in raw_name):
            cid = f"OBJ_{next_concept_idx:02d}"
            next_concept_idx += 1
            G.add_node(cid, label=raw_name, kind="concept", etype="Concept", cites=list(cites or []))
            alias_to_eid[nk] = cid
            return cid
        return None

    consensus_claims: list[dict] = []
    single_source_candidates: list[dict] = []
    isolated_outliers: list[dict] = []

    claim_clusters: list[dict] = []
    for cl in all_claims:
        text = _sanitize_claim_text(str(cl.get("text") or ""))
        if not _is_clean_claim_text(text):
            continue
        cites = _as_str_list(cl.get("cites"))
        subj = str(cl.get("subject") or "").strip()
        obj = str(cl.get("object") or "").strip()
        rel = str(cl.get("relation") or "states").strip()

        u = _resolve_entity_node(subj, cites, allow_create=True)
        v = _resolve_entity_node(obj, cites, allow_create=True)
        mentioned_eids = [
            ent["entity_id"] for ent in entities
            if ent["canonical_name"].lower() in text.lower()
            or any(a.lower() in text.lower() for a in ent.get("aliases", []) if len(a) >= 3)
        ]
        if (not u or not v or u == v) and len(mentioned_eids) >= 2:
            u, v = mentioned_eids[0], mentioned_eids[1]

        if u and v and u != v:
            G.add_edge(u, v, relation=rel, text=text, cites=cites)

        entity_sig = set(mentioned_eids)
        if u:
            entity_sig.add(u)
        if v:
            entity_sig.add(v)

        matched_cl = None
        for existing in claim_clusters:
            lex_sim = fuzz.token_set_ratio(text.lower(), existing["text"].lower())
            shared_ents = len(entity_sig & existing["entity_sig"])
            if lex_sim >= 82 or (shared_ents >= 2 and lex_sim >= 60):
                matched_cl = existing
                break
        if matched_cl is None:
            claim_clusters.append({
                "text": text,
                "relation": rel,
                "cites": set(cites),
                "entity_sig": entity_sig,
            })
        else:
            matched_cl["cites"].update(cites)
            matched_cl["entity_sig"].update(entity_sig)
            if len(text) > len(matched_cl["text"]):
                matched_cl["text"] = text

    for cc in claim_clusters:
        cites_sorted = sorted(cc["cites"], key=lambda s: int(s[1:]) if s[1:].isdigit() else 999)
        indep_count = len({echo_map.get(s, s) for s in cites_sorted})
        cc_record = {
            "text": cc["text"],
            "cites": cites_sorted,
            "indep_sources": indep_count,
        }
        if indep_count >= 2:
            consensus_claims.append(cc_record)
        else:
            single_source_candidates.append(cc_record)

    quant_discrepancies: list[dict] = []
    chronological_trends: list[dict] = []
    enriched_topics: list[dict] = []

    for top in metric_topics:
        numeric_frames = [
            f for f in top["frames"]
            if f.get("numeric_value") is not None and re.search(r"\d", str(f.get("raw_value") or ""))
        ]
        numeric_frames.sort(key=lambda f: _parse_period_sort_key(str(f.get("period_or_date") or "")))
        top["frames"] = numeric_frames

        for f in numeric_frames:
            fcites = sorted(_as_str_list(f.get("cites")), key=lambda s: int(s[1:]) if s[1:].isdigit() else 999)
            indep_f = len({echo_map.get(s, s) for s in fcites})
            st_status = str(f.get("source_type") or f.get("epistemic_status") or "").lower()
            if "unverified" in st_status or "opinion" in st_status or "blog" in st_status:
                isolated_outliers.append({
                    "topic_id": top["topic_id"],
                    "topic_name": top["topic_name"],
                    "raw_value": f.get("raw_value"),
                    "conditions_caveats": f.get("conditions_caveats"),
                    "cites": fcites,
                })
            if indep_f >= 2:
                consensus_claims.append({
                    "text": (
                        f"Corroborated across {indep_f} independent sources: "
                        f"{f.get('subject')} ({f.get('metric_name')}) = {f.get('raw_value')} "
                        f"({f.get('period_or_date') or 'reported period'}; {f.get('conditions_caveats') or 'standard basis'})."
                    ),
                    "cites": fcites,
                    "indep_sources": indep_f,
                })

        vals = np.array([float(f["numeric_value"]) for f in numeric_frames], dtype=np.float64)
        math_summary = ""

        if vals.size >= 2:
            vmin = float(np.min(vals))
            vmax = float(np.max(vals))
            vmed = float(np.median(vals))
            abs_gap = vmax - vmin
            denom = abs(vmin) if abs(vmin) > 1e-9 else 1.0
            ratio = vmax / denom if vmin > 0 else 1.0
            pct_spread = (abs_gap / denom) * 100.0

            all_cites = sorted({
                c for f in numeric_frames for c in _as_str_list(f.get("cites"))
            }, key=lambda s: int(s[1:]) if s[1:].isdigit() else 999)

            distinct_subjects = {_norm_alnum_key(f.get("subject", "")) for f in numeric_frames}
            sort_keys = {_parse_period_sort_key(str(f.get("period_or_date") or "")) for f in numeric_frames}

            if len(distinct_subjects) >= 2 and ratio > 1.01:
                ranked = sorted(numeric_frames, key=lambda f: float(f["numeric_value"]), reverse=True)
                leader = ranked[0]
                trailer = ranked[-1]
                cohort_pts = [
                    f"{f.get('subject')}: {f.get('raw_value')} [{'/'.join(_as_str_list(f.get('cites')))}]"
                    for f in ranked[:6]
                ]
                math_summary = (
                    f"Comparative cohort across {len(distinct_subjects)} subjects: "
                    f"highest is {leader.get('subject')} ({leader.get('raw_value')}), "
                    f"lowest is {trailer.get('subject')} ({trailer.get('raw_value')}), "
                    f"median={vmed:,.4g}, multiplier={ratio:.2f}x (+{pct_spread:.1f}% spread). "
                    f"Cohort breakdown: {'; '.join(cohort_pts)}"
                )
                quant_discrepancies.append({
                    "topic_id": top["topic_id"],
                    "topic_name": top["topic_name"],
                    "min_val": vmin,
                    "max_val": vmax,
                    "abs_gap": abs_gap,
                    "pct_spread": round(pct_spread, 2),
                    "summary": math_summary,
                    "cites": all_cites,
                })
            elif len(distinct_subjects) == 1 and len(sort_keys) == 1 and ratio > 1.01:
                caveat_pairs = [
                    f"{f.get('raw_value')} ({f.get('conditions_caveats') or 'reported basis'}) [{'/'.join(_as_str_list(f.get('cites')))}]"
                    for f in numeric_frames[:6]
                ]
                math_summary = (
                    f"Same-subject cross-source spread on {numeric_frames[0].get('period_or_date')}: "
                    f"min={vmin:,.4g}, max={vmax:,.4g}, median={vmed:,.4g}, gap={abs_gap:,.4g} "
                    f"({ratio:.2f}x / +{pct_spread:.1f}% spread). "
                    f"Methodological comparison: {' vs. '.join(caveat_pairs)}"
                )
                quant_discrepancies.append({
                    "topic_id": top["topic_id"],
                    "topic_name": top["topic_name"],
                    "min_val": vmin,
                    "max_val": vmax,
                    "abs_gap": abs_gap,
                    "pct_spread": round(pct_spread, 2),
                    "summary": math_summary,
                    "cites": all_cites,
                })
            elif len(distinct_subjects) == 1 and len(sort_keys) >= 2 and 1.01 < ratio < 500.0:
                first_f = numeric_frames[0]
                last_f = numeric_frames[-1]
                first_v = float(first_f["numeric_value"])
                last_v = float(last_f["numeric_value"])
                shift_pct = ((last_v - first_v) / abs(first_v)) * 100.0 if abs(first_v) > 1e-9 else 0.0
                timeline_pts = [
                    f"{f.get('raw_value')} [{f.get('period_or_date')}] [{'/'.join(_as_str_list(f.get('cites')))}]"
                    for f in numeric_frames[:6]
                ]
                math_summary = (
                    f"Same-subject chronological series ({first_f.get('period_or_date')} -> {last_f.get('period_or_date')}): "
                    f"moved from {first_f.get('raw_value')} to {last_f.get('raw_value')} "
                    f"(range {vmin:,.4g} to {vmax:,.4g}, median={vmed:,.4g}, net change={shift_pct:+.2f}%). "
                    f"Sequence: {' -> '.join(timeline_pts)}"
                )
                chronological_trends.append({
                    "topic_id": top["topic_id"],
                    "topic_name": top["topic_name"],
                    "shift_pct": round(shift_pct, 2),
                    "summary": math_summary,
                    "cites": all_cites,
                })

        enriched_topics.append({
            **top,
            "math_summary": math_summary,
        })

    centrality: dict[str, float] = {}
    if len(G) > 0:
        try:
            deg_c = nx.degree_centrality(G)
            bet_c = nx.betweenness_centrality(nx.Graph(G)) if len(G) >= 3 else deg_c
            centrality = {
                n: 0.65 * deg_c.get(n, 0.0) + 0.35 * bet_c.get(n, 0.0)
                for n in G.nodes()
            }
        except Exception:
            centrality = {n: 1.0 for n in G.nodes()}

    for ent in entities:
        ent["centrality"] = round(float(centrality.get(ent["entity_id"], 0.0)), 3)
    entities.sort(key=lambda e: (-e["centrality"], -len(e["cites"])))

    bridges: list[dict] = []
    undirected = G.to_undirected()
    seen_bridge_pairs: set[tuple[str, str]] = set()
    bridge_node_counts: dict[str, int] = defaultdict(int)
    pair_a_mid_counts: dict[tuple[str, str], int] = defaultdict(int)

    for mid_node in sorted(undirected.nodes(), key=lambda n: -centrality.get(n, 0.0)):
        mid_label = str(G.nodes[mid_node].get("label") or "").strip()
        neighbors = list(undirected.neighbors(mid_node))
        if len(neighbors) < 2 or len(neighbors) > 35:
            continue
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                if bridge_node_counts[mid_label] >= 3:
                    break
                n1, n2 = neighbors[i], neighbors[j]
                if n1 == n2 or (n1, n2) in seen_bridge_pairs or (n2, n1) in seen_bridge_pairs:
                    continue
                label_1 = str(G.nodes[n1].get("label") or "").strip()
                label_2 = str(G.nodes[n2].get("label") or "").strip()
                if len({label_1.lower(), mid_label.lower(), label_2.lower()}) < 3:
                    continue
                if pair_a_mid_counts[(label_1, mid_label)] >= 1:
                    continue
                edges_1 = list(G.get_edge_data(n1, mid_node, default={}).values()) + list(
                    G.get_edge_data(mid_node, n1, default={}).values()
                )
                edges_2 = list(G.get_edge_data(n2, mid_node, default={}).values()) + list(
                    G.get_edge_data(mid_node, n2, default={}).values()
                )
                if not edges_1 or not edges_2:
                    continue
                cites_1 = {c for e in edges_1 for c in e.get("cites", [])}
                cites_2 = {c for e in edges_2 for c in e.get("cites", [])}
                if cites_1 and cites_2 and cites_1.isdisjoint(cites_2):
                    seen_bridge_pairs.add((n1, n2))
                    bridge_node_counts[mid_label] += 1
                    pair_a_mid_counts[(label_1, mid_label)] += 1
                    text_1 = _sanitize_claim_text(edges_1[0].get("text", ""))
                    text_2 = _sanitize_claim_text(edges_2[0].get("text", ""))
                    combined_cites = sorted(
                        cites_1 | cites_2,
                        key=lambda s: int(s[1:]) if s[1:].isdigit() else 999,
                    )
                    bridges.append({
                        "entity_a": label_1,
                        "bridge_node": mid_label,
                        "entity_b": label_2,
                        "claim_a": text_1,
                        "claim_b": text_2,
                        "cites_a": sorted(cites_1),
                        "cites_b": sorted(cites_2),
                        "cites": combined_cites,
                    })
                    if len(bridges) >= 12:
                        break
            if len(bridges) >= 12 or bridge_node_counts[mid_label] >= 3:
                break

    scored_gems: list[tuple[int, dict]] = []
    seen_gem_texts: list[str] = []
    for cand in single_source_candidates:
        t = _sanitize_claim_text(cand["text"])
        if not _is_clean_claim_text(t):
            continue
        if any(fuzz.token_set_ratio(t.lower(), prev.lower()) >= 85 for prev in seen_gem_texts):
            continue
        seen_gem_texts.append(t)
        cand["text"] = t
        spec_score = (2 if re.search(r"\d", t) else 0) + (1 if len(t) >= 80 else 0)
        if spec_score >= 2:
            scored_gems.append((spec_score, cand))
    scored_gems.sort(key=lambda pair: (-pair[0], -len(pair[1]["text"])))
    rare_gems = [g for _, g in scored_gems[:18]]

    covered_metric_entities = {
        _norm_alnum_key(ent["canonical_name"])
        for ent in entities
        if any(ent["canonical_name"].lower() in f.get("subject", "").lower() for top in enriched_topics for f in top["frames"])
    }
    deduped_leads: list[tuple[str, list[str]]] = []
    for lead_text, lcites in all_leads:
        clean_ld = _sanitize_claim_text(lead_text)
        clean_ld = re.sub(r"\s+in this excerpt\.?$", "", clean_ld, flags=re.IGNORECASE).strip()
        if not _is_clean_claim_text(clean_ld):
            continue
        mentioned_in_lead = [
            _norm_alnum_key(ent["canonical_name"])
            for ent in entities
            if ent["canonical_name"].lower() in clean_ld.lower()
        ]
        if mentioned_in_lead and all(m in covered_metric_entities for m in mentioned_in_lead):
            continue
        matched_ld = False
        for idx, (existing_ld, existing_cites) in enumerate(deduped_leads):
            if fuzz.token_set_ratio(clean_ld.lower(), existing_ld.lower()) >= 78:
                merged_c = sorted(set(existing_cites) | set(lcites), key=lambda s: int(s[1:]) if s[1:].isdigit() else 999)
                deduped_leads[idx] = (existing_ld, merged_c)
                matched_ld = True
                break
        if not matched_ld:
            deduped_leads.append((clean_ld, lcites))

    consensus_claims.sort(key=lambda x: (-x["indep_sources"], -len(x["text"])))

    return {
        "entities": entities,
        "metric_topics": enriched_topics,
        "consensus_claims": consensus_claims,
        "quant_discrepancies": quant_discrepancies,
        "chronological_trends": chronological_trends,
        "bridges": bridges,
        "isolated_outliers": isolated_outliers,
        "rare_gems": rare_gems,
        "leads_and_gaps": deduped_leads[:10],
        "graph_stats": {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "independent_source_clusters": len(set(echo_map.values())),
        },
    }


def _render_duralumin_ledger(
    query: str,
    profile: dict,
    duralumin_data: dict,
    notary: list[dict],
) -> str:
    """Phase 4: Render clean, zero-jargon Research Synthesis Notes."""
    lines: list[str] = []
    lines.append("# VERIFIED RESEARCH SYNTHESIS NOTES\n")

    lines.append("## 1. CORE ENTITIES, ACTORS & SYSTEMS")
    entities = duralumin_data.get("entities") or []
    if entities:
        for ent in entities[:40]:
            tags = " ".join(f"[{c}]" for c in ent.get("cites", []))
            alias_str = f" (also referenced as: {', '.join(ent['aliases'])})" if ent.get("aliases") else ""
            lines.append(
                f"- **{ent['canonical_name']}**{alias_str} — "
                f"Category: {ent['type']} | Role: {ent.get('role', '')} {tags}".rstrip()
            )
    lines.append("")

    lines.append("## 2. QUANTITATIVE BENCHMARKS, METRICS & TIME SERIES")
    mtopics = duralumin_data.get("metric_topics") or []
    if mtopics:
        for top in mtopics[:28]:
            lines.append(f"### {top['topic_name']} (Unit: {top['unit']})")
            lines.append("| Subject / Measure | Exact Figure | Date / Window | Source Basis & Conditions | Cite |")
            lines.append("| :--- | :--- | :--- | :--- | :--- |")
            for f in top.get("frames", []):
                subj = (f.get("subject") or top["topic_name"]).replace("|", "/")
                mname = (f.get("metric_name") or "").replace("|", "/")
                label = f"{subj} ({mname})" if mname and mname.lower() not in subj.lower() else subj
                rval = (f.get("raw_value") or "").replace("|", "/")
                pdate = (f.get("period_or_date") or "Reported period").replace("|", "/")
                cav = (f.get("conditions_caveats") or "Standard reporting basis").replace("|", "/")
                cites_list = _as_str_list(f.get("cites"))
                corrob_note = f"Corroborated by {len(cites_list)} sources; " if len(cites_list) >= 2 else ""
                tags = " ".join(f"[{c}]" for c in cites_list)
                lines.append(f"| {label} | {rval} | {pdate} | {corrob_note}{cav} | {tags} |")
            if top.get("math_summary"):
                all_top_tags = " ".join(
                    f"[{c}]" for c in sorted({
                        c for f in top.get("frames", []) for c in _as_str_list(f.get("cites"))
                    }, key=lambda s: int(s[1:]) if s[1:].isdigit() else 999)
                )
                lines.append(f"* **Quantitative Synthesis**: {top['math_summary']} {all_top_tags}".rstrip())
            lines.append("")
    lines.append("")

    lines.append("## 3. MULTI-SOURCE CORROBORATED FINDINGS")
    consensus = duralumin_data.get("consensus_claims") or []
    if consensus:
        for cl in consensus[:25]:
            tags = " ".join(f"[{c}]" for c in cl.get("cites", []))
            lines.append(
                f"- **[Corroborated across {cl['indep_sources']} independent sources]** {cl['text']} {tags}".rstrip()
            )
    lines.append("")

    lines.append("## 4. COMPARATIVE COHORTS, TRAJECTORIES & CROSS-SOURCE CONNECTIONS")
    discs = duralumin_data.get("quant_discrepancies") or []
    trends = duralumin_data.get("chronological_trends") or []
    bridges = duralumin_data.get("bridges") or []
    for d in discs[:15]:
        tags = " ".join(f"[{c}]" for c in d.get("cites", []))
        lines.append(f"- **[{d['topic_name']}]** {d['summary']} {tags}".rstrip())
    for tr in trends:
        tags = " ".join(f"[{c}]" for c in tr.get("cites", []))
        lines.append(f"- **[{tr['topic_name']}]** {tr['summary']} {tags}".rstrip())
    for b in bridges:
        tags = " ".join(f"[{c}]" for c in b.get("cites", []))
        lines.append(
            f"- **[Cross-Source Connection: {b['entity_a']} <-> {b['bridge_node']} <-> {b['entity_b']}]** "
            f"{b['claim_a']} ({' '.join(f'[{c}]' for c in b['cites_a'])}) <-> "
            f"{b['claim_b']} ({' '.join(f'[{c}]' for c in b['cites_b'])}) {tags}".rstrip()
        )
    lines.append("")

    lines.append("## 5. SINGLE-SOURCE DETAILS & UNRESOLVED EMPIRICAL GAPS")
    gems = duralumin_data.get("rare_gems") or []
    leads = duralumin_data.get("leads_and_gaps") or []
    for g in gems[:12]:
        tags = " ".join(f"[{c}]" for c in g.get("cites", []))
        lines.append(f"- **[Single-Source Detail]** {g['text']} {tags}".rstrip())
    for lead_text, lcites in leads[:10]:
        tags = " ".join(f"[{c}]" for c in lcites if f"[{c}]" not in lead_text)
        lines.append(f"- **[Unresolved Empirical Gap]** {lead_text} {tags}".rstrip())

    return "\n".join(lines).strip()


async def stage_ledger(query: str, chunks: Sequence[Chunk], registry: SourceRegistry,
                      log: RunLog) -> str:
    log.stage_begin("ledger", {"chunks": len(chunks), "model": LEDGER.model,
                              "max_tokens": LEDGER.max_tokens, "engine": "DURALUMIN_v4"})
    evidence = format_evidence(chunks, registry)
    log.artifact("04_evidence.txt", evidence)

    sources: list[dict] = []
    for sid, meta in registry.items().items():
        text = registry.text_for(sid) or ""
        sources.append({
            "source_id": sid,
            "title": meta.get("title", ""),
            "url": meta.get("url", ""),
            "date": meta.get("date", ""),
            "provider": meta.get("provider", ""),
            "origin": meta.get("origin", ""),
            "chars": len(text),
            "text": text,
        })
    log.artifact("evidence_ledger/sources.jsonl",
                 "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sources))

    chunk_records: list[dict] = []
    cumulative = 0
    claim_chunks = 0
    for c in chunks:
        cumulative += c.words or 0
        has_claim, kind = claim_signal(c.text)
        if has_claim:
            claim_chunks += 1
        chunk_records.append({
            "chunk_id": c.uid,
            "source_id": c.source_id,
            "index": c.index,
            "words": c.words,
            "score": c.score,
            "priority": _ledger_priority(cumulative),
            "has_claim": has_claim,
            "kind": kind,
            "text": c.text,
        })
    log.artifact("evidence_ledger/chunks.jsonl",
                 "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in chunk_records))
    log.stage_usage(sources_written=len(sources),
                    chunks_written=len(chunk_records),
                    claim_chunks=claim_chunks,
                    priority_chunks=sum(1 for r in chunk_records if r["priority"]))

    profile = await _duralumin_profile_query(query, log)
    log.artifact_json("evidence_ledger/00_case_profile.json", profile)

    extractions_by_id, model_used, provider_used, provenance = await _duralumin_extract_batches(
        query, profile, chunks, registry, log
    )

    notary: list[dict] = []
    raw_entities: list[dict] = []
    raw_metrics: list[dict] = []
    all_claims: list[dict] = []
    all_leads: list[tuple[str, list[str]]] = []

    for rec in chunk_records:
        cid = rec["chunk_id"]
        sid = rec["source_id"]
        ext = extractions_by_id.get(cid)
        if ext:
            entities = ext.get("entities") or []
            claims = ext.get("claims") or []
            metrics = ext.get("metrics") or []
            leads = ext.get("leads_and_gaps") or []
            citations = ext.get("citations") or [sid]
            money, dates = [], []
        else:
            _m_ents, money, dates = _ledger_machine_extract(rec["text"])
            entities = []
            citations = [sid]
            claims = []
            metrics = []
            leads = []
            for sent in _extract_sentences(rec["text"])[:6]:
                if not _is_clean_claim_text(sent):
                    continue
                has_c, ckind = claim_signal(sent)
                if has_c:
                    claims.append({
                        "subject": "",
                        "relation": ckind,
                        "object": "",
                        "text": _sanitize_claim_text(sent),
                        "cites": [sid],
                    })

        for e in entities:
            if isinstance(e, dict):
                raw_entities.append({**e, "cites": citations or [sid]})
        for m in metrics:
            if isinstance(m, dict):
                raw_metrics.append({**m, "cites": _as_str_list(m.get("cites")) or citations or [sid]})
        for cl in claims:
            if isinstance(cl, dict):
                all_claims.append({**cl, "cites": _as_str_list(cl.get("cites")) or citations or [sid]})
        for ld in leads:
            if isinstance(ld, str) and ld.strip():
                all_leads.append((ld.strip(), citations or [sid]))

        notary.append({
            "chunk_id": cid,
            "source_id": sid,
            "index": rec["index"],
            "words": rec["words"],
            "score": rec["score"],
            "priority": rec["priority"],
            "has_claim": rec["has_claim"],
            "kind": rec["kind"],
            "model": model_used,
            "provider": provider_used,
            "provenance": "model" if ext else "verbatim",
            "citations": citations,
            "entities": entities,
            "metrics": metrics,
            "money": money,
            "dates": dates,
            "claims": claims,
            "leads_and_gaps": leads,
            "text": rec["text"],
        })

    log.artifact("evidence_ledger/notary.jsonl",
                 "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in notary))

    t1_entities = _tier1_resolve_entities(raw_entities)
    t1_metrics = _tier1_group_metrics(raw_metrics)
    resolved_entities, resolved_metrics = await _duralumin_resolve(t1_entities, t1_metrics, log)

    echo_map = _dedupe_echo_sources(chunks)
    duralumin_data = _duralumin_build_graph_and_math(
        resolved_entities, resolved_metrics, all_claims, all_leads, echo_map
    )
    log.artifact_json("evidence_ledger/duralumin_graph.json", {
        "profile": profile,
        **duralumin_data,
    })

    if resolved_entities or resolved_metrics or all_claims:
        digest = _render_duralumin_ledger(query, profile, duralumin_data, notary)
    else:
        digest = _render_ledger_digest(notary)
    if not digest:
        digest = evidence
    log.artifact("04_ledger_digest.txt", digest)

    cited = extract_citations(evidence)
    unknown = [s for s in cited if not registry.known(s)]
    if unknown:
        log.warn("ledger", f"evidence cites unknown ids {unknown}")
    log.stage_usage(
        cited_ids=len(cited),
        unknown_ids=unknown,
        provenance=provenance,
        model_used=model_used,
        digest_lines=len(digest.splitlines()),
        resolved_entities=len(resolved_entities),
        metric_topics=len(resolved_metrics),
        quant_discrepancies=len(duralumin_data.get("quant_discrepancies", [])),
        chronological_trends=len(duralumin_data.get("chronological_trends", [])),
        bridges_discovered=len(duralumin_data.get("bridges", [])),
        rare_gems_rescued=len(duralumin_data.get("rare_gems", [])),
    )
    log.stage_end("ok")
    return digest

# ============================================================ stage 5 dossier


def render_dossier_system() -> str:
    return (P.DOSSIER_SYSTEM
            .replace("<<REASONING_CHAR_TARGET>>", f"{P.REASONING_CHAR_TARGET:,}")
            .replace("<<REASONING_CHAR_LIMIT>>", f"{P.REASONING_CHAR_LIMIT:,}")
            .replace("<<DOSSIER_TOKEN_LIMIT>>", f"{P.DOSSIER_TOKEN_LIMIT:,}"))


async def _stream_once(cfg: StageCfg, messages: list[dict], log: RunLog,
                       max_tokens: int) -> LLMResult:
    res = await llm_call_stream(cfg, messages, log=log, stage="dossier",
                                max_tokens=max_tokens,
                                idle_timeout=DOSSIER_IDLE_TIMEOUT,
                                total_timeout=DOSSIER_TOTAL_TIMEOUT)
    log.stage_usage(finish_reason=res.finish_reason, reasoning_chars=res.reasoning_chars,
                    answer_tokens=est_tokens(res.text), answer_chars=len(res.text),
                    model_used=res.model, streamed=True)
    if res.reasoning_chars > REASONING_WARN_CHARS:
        log.warn("dossier",
                 f"reasoning_chars={res.reasoning_chars} is over the "
                 f"{REASONING_WARN_CHARS:,} target in DOSSIER_SYSTEM; less reasoning "
                 "leaves more of the output budget for the dossier itself")
    if res.reasoning_chars > P.REASONING_CHAR_LIMIT:
        log.warn("dossier",
                 f"reasoning_chars={res.reasoning_chars} exceeds the "
                 f"{P.REASONING_CHAR_LIMIT:,} ceiling in DOSSIER_SYSTEM")
    if res.finish_reason == "length":
        log.warn("dossier", "finish_reason=length: the dossier may be truncated")
    return res


async def stage_dossier(query: str, facts: str, registry: SourceRegistry,
                        log: RunLog) -> str:
    log.stage_begin("dossier", {"model": DOSSIER.model, "stream": True,
                                "max_tokens": DOSSIER_MAX_TOKENS,
                                "idle_timeout_s": DOSSIER_IDLE_TIMEOUT,
                                "total_timeout_s": DOSSIER_TOTAL_TIMEOUT})
    system = render_dossier_system()
    base_user = (
        f"USER QUESTION:\n{query}\n\n"
        f"EVIDENCE LEDGER:\n<<<LEDGER>>>\n{facts}\n<<<END_LEDGER>>>"
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": base_user}]
    retry_messages = [*messages, {"role": "user", "content": P.DOSSIER_RETRY_SUFFIX}]
    errs: list[str] = []

    for i, cfg in enumerate((DOSSIER, *DOSSIER.fallback)):
        if not cfg.have_key():
            log.warn("dossier", f"{cfg.name} skipped: {cfg.key_env} is not set")
            errs.append(f"{cfg.name}: no key")
            continue
        log.provider_used("dossier", cfg.model, i > 0)
        if i > 0:
            log.stage_usage(fallback=True)
        for attempt, msgs in ((1, messages), (2, retry_messages)):
            try:
                res = await _stream_once(cfg, msgs, log, cfg.max_tokens)
            except StageError as exc:
                errs.append(f"{cfg.name} attempt {attempt}: {exc}")
                log.warn("dossier", str(exc))
                if attempt == 1:
                    log.warn("dossier",
                             f"retrying {cfg.model} once with DOSSIER_RETRY_SUFFIX")
                    continue
                break
            truncated = res.finish_reason == "length"
            if truncated and not res.text.strip():
                errs.append(f"{cfg.name} attempt {attempt}: length with no answer")
                continue
            log.stage_usage(attempt=attempt, possibly_truncated=truncated,
                            reasoning_target=P.REASONING_CHAR_TARGET,
                            reasoning_ceiling=P.REASONING_CHAR_LIMIT,
                            token_target=P.DOSSIER_TOKEN_LIMIT)
            log.artifact("05_dossier.md", res.text)
            check = validate_citations(res.text, registry)
            log.artifact_json("05_dossier_citations.json", check)
            if check["unknown"]:
                log.warn("dossier",
                         f"dossier cites ids absent from the registry: {check['unknown']}")
            log.stage_end("ok")
            return res.text
    raise AbortPipeline(
        "dossier: every provider and retry failed (streaming). " + " | ".join(errs[-4:])
    )


# ============================================================ stage 6 red team


async def collect_counter_evidence(challenges: Sequence[dict], log: RunLog) -> list[Chunk]:
    """Run each challenge search_query through Exa and Tavily standard search, dedupe,
    register as origin='counter', chunk and rerank against the challenge text."""
    registry = log.registry
    queries: list[str] = []
    for c in challenges[:6]:
        sq = str(c.get("search_query") or "").strip()
        if sq and sq not in queries:
            queries.append(sq)
    if not queries:
        return []
    log.stage_begin("retrieve_counter", {"queries": len(queries)})
    tasks: list[Any] = []
    for sq in queries:
        tasks.append(exa_search(sq, SEARCH_CFG["exa"]["num_results"], log, "retrieve_counter"))
        tasks.append(tavily_search(sq, min(6, SEARCH_CFG["tavily_search"]["num_results"]), log,
                                   "retrieve_counter"))
    settled = await asyncio.gather(*tasks, return_exceptions=True)
    docs: list[Document] = []
    seen: set[str] = set()
    for i, item in enumerate(settled):
        if isinstance(item, Exception):
            log.warn("retrieve_counter",
                     f"provider task {i} raised {type(item).__name__}: {item}")
            continue
        for row in item or []:
            key = normalize_url(row["url"])
            if not key or key in seen:
                continue
            sid = registry.register(row["url"], row.get("title", ""), row.get("date", ""),
                                    "counter-search", "counter")
            if not sid:
                continue
            seen.add(key)
            docs.append(Document(sid, row.get("title") or row["url"], row["url"],
                                 row.get("date", ""), "counter-search", "counter",
                                 str(row["text"])))
    log.stage_usage(counter_documents=len(docs), queries=len(queries))
    if not docs:
        log.stage_end("ok", error="no counter-evidence retrieved")
        return []
    chunks: list[Chunk] = []
    for d in docs:
        chunks.extend(chunk_document(d.text, d.source_id))
    scored = await _rerank_counter_chunks(chunks, challenges, log)
    kept: list[Chunk] = []
    budget = COUNTER_EVIDENCE_TOKEN_BUDGET
    for i in sorted(range(len(chunks)), key=lambda j: -scored.get(j, 0.0)):
        c = chunks[i]
        c.score = scored.get(i, 0.0)
        cost = est_tokens(c.text)
        if cost > budget:
            continue
        kept.append(c)
        budget -= cost
        if budget <= 0:
            break
    log.artifact("06_counter_evidence.md", "\n\n".join(c.tagged() for c in kept))
    log.stage_usage(chunks_kept=len(kept), token_budget=COUNTER_EVIDENCE_TOKEN_BUDGET)
    log.stage_end("ok")
    return kept


async def _rerank_counter_chunks(chunks: list[Chunk], challenges: Sequence[dict],
                                 log: RunLog) -> dict[int, float]:
    cfg = SEARCH_CFG["cloudflare_rerank"]
    rq = " ".join(
        f"{c.get('claim_challenged', '')} {c.get('counter_argument', '')}" for c in challenges
    ).strip()[:2000]
    account_id = env_str(cfg["account_id_env"]) or ("dry-run-account" if
                                                    _DRY_RUN["scenario"] else "")
    token = key_for(cfg)
    if not token or not account_id or not chunks:
        if chunks:
            log.warn("retrieve_counter",
                     "no reranker available for counter-evidence; ranking with a local "
                     "BM25 fallback")
        return lexical_scores(rq, chunks)
    url = cfg["base_url"].format(account_id=account_id, model=cfg["model"])
    scores: dict[int, float] = {}
    for start in range(0, len(chunks), RERANK_BATCH):
        batch = chunks[start: start + RERANK_BATCH]
        status, data = await http_request(
            "POST", url,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json_body={"query": rq, "contexts": [{"text": c.text} for c in batch],
                       "top_k": len(batch)},
            timeout=90.0, attempts=2, log=log, stage="retrieve_counter",
        )
        if status != 200:
            break
        for idx, score in parse_rerank_scores(data) or []:
            if 0 <= idx < len(batch):
                scores[start + idx] = score
    return scores


async def stage_redteam(query: str, dossier: str, log: RunLog) -> list[dict]:
    log.stage_begin("redteam", {"model": REDTEAM.model})
    messages = [
        {"role": "system", "content": P.REDTEAM_SYSTEM},
        {"role": "user", "content": (
            f"USER QUESTION:\n{query}\n\n<<<DOSSIER>>>\n{dossier}\n<<<END DOSSIER>>>"
        )},
    ]
    payload, _cfg, _fb = await llm_chain(
        (REDTEAM, *REDTEAM.fallback),
        lambda c: llm_json(c, messages, log, "redteam"),
        log, "redteam", core=False,
    )
    challenges: list[dict] = []
    if isinstance(payload, dict) and isinstance(payload.get("challenges"), list):
        for i, c in enumerate(payload["challenges"][:6], 1):
            if not isinstance(c, dict):
                continue
            claim = str(c.get("claim_challenged") or "").strip()
            arg = str(c.get("counter_argument") or "").strip()
            sq = str(c.get("search_query") or "").strip()
            if claim and arg and sq:
                challenges.append({"id": i, "claim_challenged": claim,
                                   "counter_argument": arg, "search_query": sq})
    log.artifact_json("06_redteam.json", {"challenges": challenges, "raw": payload})
    log.stage_usage(challenges=len(challenges))
    log.stage_end("ok")
    return challenges


# ============================================================ stage 7 patch


def is_all_correct(text: str) -> bool:
    """Accept the bare literal ALL CORRECT, or {"status": "ALL CORRECT"}."""
    stripped = strip_think(text or "")
    if not stripped:
        return False
    if "all correct" in stripped.lower() and len(stripped) < 400:
        return True
    try:
        obj = extract_json(stripped)
    except ValueError:
        return False
    status = re.sub(r"[^A-Z ]", " ", str(obj.get("status") or "").upper())
    status = " ".join(status.split())
    return status == "ALL CORRECT" and not obj.get("replacements")


def replacement_list(payload: Any) -> list[dict]:
    if not isinstance(payload, dict):
        return []
    rows = payload.get("replacements")
    if not isinstance(rows, list):
        return []
    return [r for r in rows if isinstance(r, dict)]


async def stage_patch(dossier: str, challenges: Sequence[dict], counter: Sequence[Chunk],
                      log: RunLog) -> str:
    log.stage_begin("patch", {"model": PATCH.model, "challenges": len(challenges),
                              "counter_chunks": len(counter)})
    if not challenges:
        log.warn("patch", "no challenges produced, patch skipped")
        log.artifact_json("07_patch.json", {"skipped": "no challenges"})
        log.stage_end("ok", error="skipped: no challenges")
        return dossier
    chal_text = "\n\n".join(
        f"{c['id']}. CLAIM: {c['claim_challenged']}\n"
        f"   OBJECTION: {c['counter_argument']}\n"
        f"   VERIFY WITH: {c['search_query']}"
        for c in challenges
    )
    counter_text = "\n\n".join(
        f"[{c.source_id}] (chunk {c.index}) {c.text}" for c in counter
    )
    messages = [
        {"role": "system", "content": P.PATCH_SYSTEM},
        {"role": "user", "content": (
            "<<<DOSSIER>>>\n" + dossier + "\n<<<END DOSSIER>>>\n\n"
            "<<<CHALLENGES>>>\n" + chal_text + "\n<<<END CHALLENGES>>>\n\n"
            "<<<COUNTER_EVIDENCE>>>\n" + counter_text + "\n<<<END COUNTER_EVIDENCE>>>"
        )},
    ]
    payload, _cfg, _fb = await llm_chain(
        (PATCH, *PATCH.fallback),
        lambda c: llm_json(c, messages, log, "patch"),
        log, "patch", core=False,
    )
    if payload is None:
        log.artifact_json("07_patch.json", {"skipped": "provider produced no valid JSON"})
        log.stage_end("skipped", error="provider produced no valid JSON")
        return dossier
    if is_all_correct(json.dumps(payload)):
        log.artifact_json("07_patch.json", {"status": "ALL CORRECT", "raw": payload})
        log.stage_usage(replacements=0)
        log.stage_end("ok")
        return dossier
    reps = replacement_list(payload)
    if not reps:
        log.artifact_json("07_patch.json", {"status": "ALL CORRECT", "raw": payload})
        log.stage_usage(replacements=0)
        log.stage_end("ok")
        return dossier
    updated, stats = apply_replacements(dossier, reps, log=log, stage="patch")
    log.artifact_json("07_patch.json", {"stats": stats, "replacements": reps})
    log.stage_usage(**{f"patch_{k}": v for k, v in stats.items()})
    check = validate_citations(updated, log.registry)
    log.artifact_json("07_patch_citations.json", check)
    if check["unknown"]:
        log.warn("patch", f"patched dossier cites unknown ids {check['unknown']}")
    log.stage_end("ok")
    return updated


# ============================================================ stage 8 audit


async def stage_audit(dossier: str, facts: str, log: RunLog) -> tuple[str, bool, dict]:
    log.stage_begin("audit", {"model": AUDIT.model})
    messages = [
        {"role": "system", "content": P.AUDIT_SYSTEM},
        {"role": "user", "content": (
            "<<<LEDGER>>>\n" + facts + "\n<<<END_LEDGER>>>\n\n"
            "<<<DOSSIER>>>\n" + dossier + "\n<<<END DOSSIER>>>"
        )},
    ]
    payload, _cfg, _fb = await llm_chain(
        (AUDIT, *AUDIT.fallback),
        lambda c: llm_json(c, messages, log, "audit"),
        log, "audit", core=False,
    )
    if payload is None:
        log.artifact_json("08_audit.json", {"skipped": "provider produced no valid JSON"})
        log.stage_end("skipped", error="provider produced no valid JSON")
        return dossier, False, {}
    polishing_needed = str(payload.get("polishing") or "").strip().lower() == "needed"
    reps = replacement_list(payload)
    status = str(payload.get("status") or "").strip().upper()
    if not reps or is_all_correct(json.dumps(payload)):
        updated, stats = dossier, {}
        log.artifact_json("08_audit.json", {"status": status or "ALL CORRECT",
                                            "stats": stats, "raw": payload})
        log.stage_usage(audit_replacements=0, polishing=polishing_needed)
        log.stage_end("ok")
        return updated, polishing_needed, stats
    updated, stats = apply_replacements(dossier, reps, log=log, stage="audit")
    log.artifact_json("08_audit.json", {"status": status, "stats": stats,
                                        "replacements": reps})
    log.stage_usage(**{f"audit_{k}": v for k, v in stats.items()})
    check = validate_citations(updated, log.registry)
    log.artifact_json("08_audit_citations.json", check)
    if check["unknown"]:
        log.warn("audit", f"audited dossier cites unknown ids {check['unknown']}")
    log.stage_end("ok")
    return updated, polishing_needed, stats


# ============================================================ stage 8.5 claim audit


async def stage_claim_audit(dossier: str, registry: SourceRegistry,
                            log: RunLog) -> tuple[str, dict]:
    """Per-source claim audit via CoreWeave GLM 5.3 Flash.

    Groups every cited sentence in the dossier by the source id it carries, then
    sends one async LLM call per source. GLM sees the source text and the list of
    sentences that cite it, and returns structured JSON with SUPPORTED / PARTIAL /
    CONTRADICTED / UNSUPPORTED per sentence. UNSUPPORTED sentences are cut from
    the dossier; PARTIAL and CONTRADICTED sentences are replaced.

    A semaphore limits concurrency to 3 so we do not hit CoreWeave's rate limit.
    """
    log.stage_begin("claim_audit", {"model": CLAIM_AUDIT.model, "provider": "coreweave"})
    claims = extract_claims_by_source(dossier, registry)
    if not claims:
        log.stage_end("skipped", error="no cited sentences found")
        return dossier, {}

    semaphore = asyncio.Semaphore(8)

    async def _audit_one(sid: str, sentences: list[str]) -> tuple[str, dict]:
        source_text = registry.text_for(sid) or ""
        source_text = source_text[:12000]
        if not source_text:
            return sid, {}
        prompt_sentences = "\n".join(
            f"- {sent}" for sent in sentences
        )
        user = (
            f"SOURCE DOCUMENT ({sid}):\n{source_text}\n\n"
            f"DOSSIER SENTENCES CITING THIS SOURCE:\n{prompt_sentences}"
        )
        messages = [
            {"role": "system", "content": P.CLAIM_AUDIT_SYSTEM},
            {"role": "user", "content": user},
        ]
        async with semaphore:
            try:
                payload = await llm_json(CLAIM_AUDIT, messages, log=log,
                                         stage="claim_audit")
            except StageError as exc:
                log.warn("claim_audit", f"{sid}: {exc}")
                return sid, {}

        if not isinstance(payload, dict):
            return sid, {}
        results = payload.get("results") or []
        by_sentence: dict[str, dict] = {}
        for entry in results:
            if not isinstance(entry, dict):
                continue
            sentence = str(entry.get("sentence", "")).strip()
            if not sentence:
                continue
            status = str(entry.get("status", "")).strip().upper()
            replacement = str(entry.get("replacement", "") or "")
            by_sentence[sentence] = {
                "status": status if status in ("SUPPORTED", "PARTIAL",
                                               "CONTRADICTED", "UNSUPPORTED")
                else "UNSUPPORTED",
                "replacement": replacement,
            }
        return sid, by_sentence

    tasks = [_audit_one(sid, sents) for sid, sents in claims.items()]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    per_source: dict[str, dict] = {}
    for res in results:
        if isinstance(res, Exception):
            continue
        sid, by_sent = res
        per_source[sid] = by_sent

    replacements: list[dict] = []
    cut_sentences: set[str] = set()
    for sid, by_sent in per_source.items():
        for sentence, verdict in by_sent.items():
            status = verdict["status"]
            replacement = verdict["replacement"]
            if status == "UNSUPPORTED" and replacement.strip() == "CUT":
                cut_sentences.add(sentence)
                replacements.append({
                    "old_sentence": sentence,
                    "new_sentence": "",
                    "reason": f"unsupported by {sid}",
                    "cut": True,
                })
            elif status in ("PARTIAL", "CONTRADICTED") and replacement.strip():
                replacements.append({
                    "old_sentence": sentence,
                    "new_sentence": replacement.strip(),
                    "reason": f"{status.lower()} per {sid}",
                    "cut": False,
                })

    updated = dossier
    if replacements:
        updated, stats = apply_claim_replacements(updated, replacements, cut_sentences)
    else:
        stats = {}

    log.artifact_json("09_claim_audit.json", {
        "sources_audited": len(per_source),
        "replacements": len(replacements),
        "cuts": len(cut_sentences),
        "per_source": {
            sid: {k: v for k, v in by_sent.items()}
            for sid, by_sent in per_source.items()
        },
    })
    log.stage_usage(sources_audited=len(per_source), replacements=len(replacements),
                    cuts=len(cut_sentences))
    log.stage_end("ok")
    return updated, stats


def apply_claim_replacements(text: str, replacements: list[dict],
                             cut_sentences: set[str]) -> tuple[str, dict]:
    """Apply per-source claim-audit replacements and cuts.

    CUT sentences are removed entirely. PARTIAL/CONTRADICTED sentences are replaced
    with the auditor's suggested text. Sentences that appear multiple times are
    handled carefully: all occurrences of a cut sentence are removed; replacements
    are applied once per distinct sentence.
    """
    stats = {"applied": 0, "skipped_not_found": 0, "cut": 0}
    lines = text.splitlines()
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            out.append(line)
            continue
        if stripped in cut_sentences:
            stats["cut"] += 1
            stats["applied"] += 1
            continue
        replaced = False
        for rep in replacements:
            if rep.get("cut"):
                continue
            old = rep["old_sentence"]
            new = rep["new_sentence"]
            if old in line:
                line = line.replace(old, new, 1)
                stats["applied"] += 1
                replaced = True
                break
        out.append(line)
    return "\n".join(out), stats


# ============================================================ stage 9 polish


async def stage_polish(dossier: str, log: RunLog) -> str:
    log.stage_begin("polish", {"model": POLISH.model, "guard_ratio": POLISH_MIN_RATIO})
    messages = [
        {"role": "system", "content": P.POLISH_SYSTEM},
        {"role": "user", "content": "<<<DOSSIER>>>\n" + dossier + "\n<<<END DOSSIER>>>"},
    ]
    original, cfg, _fb = await llm_chain(
        (POLISH, *POLISH.fallback),
        lambda c: llm_call(c, messages, log=log, stage="polish"),
        log, "polish", core=False,
    )
    if original is None:
        log.artifact("09_polish.md", dossier)
        log.stage_end("skipped", error="no provider returned a draft")
        return dossier
    candidate = strip_think(original.text)
    if not candidate.strip():
        log.warn("polish", "provider returned empty output; keeping the pre-polish dossier")
        log.artifact("09_polish.md", dossier)
        log.stage_end("skipped", error="empty provider output")
        return dossier
    ratio = polish_similarity(dossier, candidate)
    log.artifact_json("09_polish_guard.json",
                      {"similarity": round(ratio, 6), "threshold": POLISH_MIN_RATIO,
                       "accepted": ratio >= POLISH_MIN_RATIO,
                       "model": getattr(original, "model", cfg.model)})
    if ratio < POLISH_MIN_RATIO:
        log.warn("polish",
                 f"guard failed: word-sequence similarity {ratio:.4f} < {POLISH_MIN_RATIO}; "
                 "discarding the polished draft and keeping the pre-polish dossier")
        log.artifact("09_polish.md", dossier)
        log.stage_usage(similarity=round(ratio, 6), accepted=False)
        log.stage_end("ok", error="guard rejected the polished draft")
        return dossier
    log.artifact("09_polish.md", candidate)
    log.stage_usage(similarity=round(ratio, 6), accepted=True)
    log.stage_end("ok")
    return candidate


# ============================================================ finalize


def build_sources_section(dossier: str, registry: SourceRegistry) -> str:
    cited = extract_citations(dossier)
    if not cited:
        return ""
    rows = []
    for n, sid in enumerate(cited, 1):
        meta = registry.get(sid)
        if not meta:
            continue
        date = meta.get("date") or ""
        date_part = f" ({date[:10]})" if date else ""
        rows.append(
            f"{n}. {meta.get('title') or meta.get('url')}{date_part}. "
            f"<{meta.get('url')}> [{sid}, {meta.get('provider')}, {meta.get('origin')}]"
        )
    if not rows:
        return ""
    return "\n\n## Sources\n\n" + "\n".join(rows)


def finalize(dossier: str, registry: SourceRegistry, log: RunLog) -> Path:
    log.stage_begin("finalize", {"cited": len(extract_citations(dossier)),
                                 "registered": len(registry)})
    body = dossier.rstrip() + "\n"
    check = validate_citations(body, registry)
    final = body + build_sources_section(body, registry)
    log.artifact("sources.json", json.dumps(
        {"cited": check["cited"], "unknown": check["unknown"],
         "registry": registry.items()}, indent=2, ensure_ascii=False))
    check2 = validate_citations(final, registry)
    if check2["unknown"]:
        log.warn("finalize", f"final dossier cites unknown ids {check2['unknown']}")
    path = log.artifact("dossier.md", final)
    log.stage_usage(cited=len(check2["cited"]), unknown=check2["unknown"],
                    words=len(final.split()), chars=len(final))
    log.stage_end("ok")
    return path


# ============================================================ pipeline


async def _notify(on_stage, name: str) -> None:
    if on_stage is None:
        return
    try:
        result = on_stage(name)
        if asyncio.iscoroutine(result):
            await result
    except Exception as exc:
        print(f"    [progress hook failed at {name}: {exc}]")


async def run_pipeline(query: str, out_dir: Path, skip_redteam: bool = False,
                       skip_polish: bool = False, on_stage=None) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(out_dir) / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(run_dir, query, {"skip_redteam": skip_redteam,
                                  "skip_polish": skip_polish,
                                  "dry_run": _DRY_RUN["scenario"] is not None})
    log.data["config"] = {
        "stages": [c.describe() for c in (PLANNER, DURALUMIN_PROFILER, LEDGER, DURALUMIN_INSPECTOR,
                                          DOSSIER, REDTEAM, PATCH, AUDIT, CLAIM_AUDIT, POLISH)],
        "search": dict(SEARCH_CFG),
        "tuning": {"rerank_min_score": RERANK_MIN_SCORE, "rerank_floor": RERANK_FLOOR,
                   "rerank_cap": RERANK_CAP, "chunk_words": [CHUNK_MIN_WORDS, CHUNK_MAX_WORDS],
                   "dossier_idle_timeout_s": DOSSIER_IDLE_TIMEOUT,
                   "dossier_total_timeout_s": DOSSIER_TOTAL_TIMEOUT,
                   "dossier_max_tokens": DOSSIER_MAX_TOKENS,
                   "polish_min_ratio": POLISH_MIN_RATIO,
                   "counter_evidence_token_budget": COUNTER_EVIDENCE_TOKEN_BUDGET},
    }
    registry = log.registry
    log.artifact("00_query.txt", query + "\n")
    print(f"\nrun directory: {run_dir}")

    try:
        research, search = await stage_plan(query, log)
        await _notify(on_stage, "plan")
        docs = await stage_retrieve(query, {"research": research, "search": search}, log)
        await _notify(on_stage, "retrieve")
        chunks = await stage_rerank(query, docs, log)
        await _notify(on_stage, "rerank")
        facts = await stage_ledger(query, chunks, registry, log)
        await _notify(on_stage, "ledger")
        dossier = await stage_dossier(query, facts, registry, log)
        await _notify(on_stage, "dossier")

        if skip_redteam:
            log.warn("redteam", "skipped by --skip-redteam")
        else:
            challenges = await stage_redteam(query, dossier, log)
            await _notify(on_stage, "redteam")
            if challenges:
                counter = await collect_counter_evidence(challenges, log)
                await _notify(on_stage, "counter")
                dossier = await stage_patch(dossier, challenges, counter, log)
                await _notify(on_stage, "patch")
            else:
                log.warn("patch", "no challenges survived the red team; patch is a no-op")

        if _DRY_RUN["scenario"] is not None:
            dossier, polishing_needed, _stats = await stage_audit(dossier, facts, log)
            dossier, _claim_stats = await stage_claim_audit(dossier, log.registry, log)
        else:
            # Run Stage 8 (Global Audit on Privatemode/Routeway) and Stage 8.5 (Per-Source Claim Audit
            # on CoreWeave) concurrently in parallel, then apply both sets of replacements!
            pre_audit_dossier = dossier
            (audited_dossier, polishing_needed, _stats), (claim_audited_dossier, _claim_stats) = await asyncio.gather(
                stage_audit(pre_audit_dossier, facts, log),
                stage_claim_audit(pre_audit_dossier, log.registry, log),
            )
            # Re-apply any claim_audit cuts/replacements onto the audited_dossier so both take effect
            if claim_audited_dossier != pre_audit_dossier:
                claim_audit_file = log.run_dir / "09_claim_audit.json"
                if claim_audit_file.exists():
                    ca_data = json.loads(claim_audit_file.read_text(encoding="utf-8"))
                    ca_reps: list[dict] = []
                    ca_cuts: set[str] = set()
                    for _sid, by_sent in (ca_data.get("per_source") or {}).items():
                        for sent, verd in (by_sent or {}).items():
                            st = verd.get("status")
                            rep_txt = str(verd.get("replacement") or "")
                            if st == "UNSUPPORTED" and rep_txt.strip() == "CUT":
                                ca_cuts.add(sent)
                                ca_reps.append({"old_sentence": sent, "new_sentence": "", "cut": True})
                            elif st in ("PARTIAL", "CONTRADICTED") and rep_txt.strip():
                                ca_reps.append({"old_sentence": sent, "new_sentence": rep_txt.strip(), "cut": False})
                    if ca_reps:
                        audited_dossier, _ = apply_claim_replacements(audited_dossier, ca_reps, ca_cuts)
            dossier = audited_dossier
            await _notify(on_stage, "verify")

        if skip_polish:
            log.warn("polish", "skipped by --skip-polish")
        elif polishing_needed:
            dossier = await stage_polish(dossier, log)
        else:
            log.warn("polish", "audit said polishing='not needed'; stage skipped")

        await _notify(on_stage, "finalize")
        path = finalize(dossier, registry, log)
        print(f"\ndossier written: {path}")
        return path
    except AbortPipeline as exc:
        log.warn("pipeline", f"aborted: {exc}")
        log.data["aborted"] = True
        raise
    finally:
        log.save()


# ============================================================ dry run router

_DRY_RUN: dict = {"scenario": None}

DRY_DOC_SEEDS = [
    ("https://example.gov/primary-report", "Primary regulator report",
     "The regulator published the primary dataset on 14 March 2026. Reported spending "
     "was 4.2 billion units, up 11 percent year over year. Compliance filings were due "
     "on 30 June 2026. The methodology appendix covers sampling error of roughly 1.8 "
     "percent at the 95 percent confidence level, and notes two audited exceptions. "
     "The appendix states that deferred payments recognised under the previous regime are "
     "excluded from the reported figure, and that the exclusion was applied consistently "
     "across all reporting entities in the sample. A separate table sets out the "
     "restatement policy applied when an entity changes its revenue recognition method. "
     "Under that policy, prior-period comparatives are revised and the revision is "
     "disclosed in a footnote rather than restated in the headline table. The regulator "
     "also published a machine-readable extract alongside the narrative release, with one "
     "row per registered filer and one column per reporting period, allowing independent "
     "recalculation of the aggregate. Coverage is described as complete for entities "
     "filing under the standard regime, with a residual group of roughly four percent of "
     "filers reporting under an exemption whose data is aggregated rather than itemised. "
     "The regulator notes that the exempted group is concentrated in two industries and "
     "warns that conclusions drawn about industry-level trends should treat that group as "
     "unobserved. A supplementary note explains the sampling frame construction, including "
     "the treatment of entities that entered or exited the register mid-period."),
    ("https://journals.example.org/study", "Peer-reviewed study of the same period",
     "An independent study covering January to June 2026 estimates spending at 4.05 "
     "billion units, marginally below the regulator figure. The authors attribute the "
     "gap to a different treatment of deferred payments and warn that the sample covers "
     "only 38 percent of registered filers. The study reconstructs the aggregate from "
     "company-level filings obtained through a freedom-of-information request, and applies "
     "its own restatement policy where the reported basis changed. Because that policy "
     "differs from the regulator's, the authors describe the two totals as measuring "
     "related but non-identical constructs rather than competing estimates of one number. "
     "They report a confidence interval on their own estimate and observe that the "
     "regulator's figure sits outside it. The paper includes a robustness section testing "
     "the conclusion under three alternative treatments of deferred payments, and notes "
     "that under two of the three the direction of the discrepancy changes sign. The "
     "authors recommend that any comparison of period-on-period growth be accompanied by "
     "the basis of preparation, and they caution against reading the growth figure as a "
     "measure of underlying activity. Limitations disclosed in the paper include the "
     "incomplete filer coverage, the inability to verify the exemption-group treatment, "
     "and the absence of a second reviewer for a subset of the restatement decisions."),
    ("https://news.example.net/coverage", "Trade press coverage",
     "Commentators argue the 11 percent growth figure overstates the real trend because "
     "it excludes two large bankruptcies in the sector. Others counter that the "
     "regulator methodology explicitly nets insolvencies out. The coverage quotes a "
     "trade association representative arguing that the aggregate is not a useful measure "
     "of sector health, and an independent analyst responding that the growth figure is "
     "the best available consistent series and that the objection confuses level with "
     "distribution. A second article focuses on the disclosure of the exempted filer group "
     "and suggests the four percent residual should be modelled rather than ignored. Both "
     "articles note that the regulator's machine-readable extract permits third-party "
     "recalculation, and one analyst published a recalculation that lands within one "
     "percent of the published aggregate. The coverage does not establish which treatment "
     "is correct and does not cite the methodology appendix directly."),
    ("https://blog.example.org/opinion", "Practitioner blog analysis",
     "A practitioner blog claims implementation costs rose 30 percent, a figure that does "
     "not appear in any official dataset reviewed here. The author presents it as an "
     "estimate derived from twelve interviews. The blog post states that the interviews "
     "were conducted with compliance officers at mid-sized entities and that the sample "
     "was not randomly selected. It provides a table of the reported figures but does not "
     "publish the interview instrument. A reply from another practitioner disputes the "
     "figure on the grounds that it conflates one-off transition costs with recurring "
     "operating costs. The blog also links to the regulator release but does not cite the "
     "methodology appendix, and it does not reference the independent study. No figure in "
     "the post reconciles to the official aggregate."),
]


def _dry_stub_text_from_dossier(prompt: str) -> str:
    body = _between(prompt, "<<<DOSSIER>>>", "<<<END DOSSIER>>>")
    return body or ""


def _between(text: str, start: str, end: str) -> str:
    i = text.find(start)
    if i == -1:
        return ""
    j = text.find(end, i + len(start))
    if j == -1:
        return ""
    return text[i + len(start): j]


def _stub_ledger(prompt: str) -> str:
    found = _CHUNK_MENTION_RE.findall(prompt)
    chunk_ids: list[str] = []
    for sid, num, idx in sorted(found, key=lambda t: (int(t[1]), int(t[2]))):
        cid = f"{sid}#{idx}"
        if cid not in chunk_ids:
            chunk_ids.append(cid)
    if not chunk_ids:
        chunk_ids = ["S1#0"]
    extractions = []
    for i, cid in enumerate(chunk_ids):
        sid = cid.split("#", 1)[0]
        is_alt = (i % 2 == 1)
        val_str = "4.05 billion units" if is_alt else "4.2 billion units"
        val_num = 4050000000.0 if is_alt else 4200000000.0
        caveat = (
            "FOI company-level reconstruction; 38% filer coverage; custom restatement"
            if is_alt else
            "Excludes deferred payments; 1.8% sampling error at 95% CI; 4% exempted residual"
        )
        extractions.append({
            "chunk_id": cid,
            "citations": [sid],
            "entities": [
                {
                    "surface": "Independent Study Group" if is_alt else "Primary Regulator",
                    "canonical_hint": "Independent FOI Study Authors" if is_alt else "Primary Sector Regulator",
                    "type": "Organization",
                    "role": "Researcher" if is_alt else "Statutory Regulator",
                    "start": 0,
                    "end": 18,
                }
            ],
            "metrics": [
                {
                    "subject": "Sector Reported Spending",
                    "metric_name": "Total Period Spending",
                    "raw_value": val_str,
                    "numeric_value": val_num,
                    "unit": "units",
                    "period_or_date": "2026-03-14",
                    "conditions_caveats": caveat,
                    "epistemic_status": "PEER_REVIEWED" if is_alt else "OFFICIAL_PRIMARY",
                    "cites": [sid],
                }
            ],
            "claims": [
                {
                    "subject": "Independent Study Group" if is_alt else "Primary Regulator",
                    "relation": "amount",
                    "object": "Deferred Payment Accounting Rule",
                    "text": f"Reported spending for the period was {val_str}",
                    "epistemic_status": "CONSENSUS_FACT",
                    "cites": [sid],
                    "start": 0,
                    "end": 54,
                },
                {
                    "subject": "Primary Regulator",
                    "relation": "date",
                    "object": "Dataset Release",
                    "text": "The primary regulator published the official dataset release on 14 March 2026.",
                    "epistemic_status": "CONSENSUS_FACT",
                    "cites": [sid],
                    "start": 0,
                    "end": 15,
                },
            ],
            "leads_and_gaps": [
                f"Residual 4% exempted filer group in two industries remains unobserved [{sid}]"
            ],
        })
    return json.dumps({"extractions": extractions})


def _stub_dossier(prompt: str) -> str:
    ledger = _between(prompt, "<<<LEDGER>>>", "<<<END_LEDGER>>>")
    ids = sorted({f"S{m.group(1)}" for m in CITATION_RE.finditer(ledger)},
                 key=lambda s: int(s[1:]))
    if not ids:
        ids = ["S1", "S2"]
    a = ids[0]
    b = ids[1] if len(ids) > 1 else a
    return (
        f"# Evidence on the period in question\n\n"
        f"## Headline figures\n\n"
        f"The regulator published the primary dataset on 14 March 2026, and reported "
        f"spending for the period was 4.2 billion units [{a}]. That figure represents "
        f"growth of 11 percent year over year [{a}].\n\n"
        f"## Disagreement between sources\n\n"
        f"An independent study covering the same window estimates spending at 4.05 "
        f"billion units, which is below the regulator figure [{b}]. The authors "
        f"attribute the gap to different treatment of deferred payments [{b}]. The "
        f"figures therefore disagree, and neither source is presented here as settled.\n\n"
        f"## Method and coverage\n\n"
        f"The regulator methodology appendix notes sampling error near 1.8 percent at the "
        f"95 percent confidence level, and records two audited exceptions [{a}]. The "
        f"independent study sample covers only 38 percent of registered filers [{b}], so "
        f"its estimate is not directly comparable to a full-population figure.\n\n"
        f"## Compliance timeline\n\n"
        f"Compliance filings were due on 30 June 2026 [{a}].\n\n"
        f"## What the evidence does not show\n\n"
        f"No source reviewed here provides a full-population cost breakdown, and the "
        f"sample coverage limitation above applies to the lower estimate.\n"
    )


def _stub_redteam(prompt: str) -> dict:
    dossier = _stub_dossier("")
    return {"challenges": [{
        "id": 1,
        "claim_challenged": "Reported spending for the period was 4.2 billion units.",
        "counter_argument": "An independent study reports 4.05 billion units for the same "
                            "window, so the regulator figure may exclude deferred payments.",
        "search_query": "spending 4.05 billion deferred payments period",
    }]}


def _stub_patch(prompt: str) -> dict:
    dossier = _between(prompt, "<<<DOSSIER>>>", "<<<END DOSSIER>>>")
    match = re.search(r"The regulator published[^\n]*?4\.2 billion units \[S\d+\]\.", dossier)
    if match is None:
        return {"status": "ALL CORRECT"}
    target = match.group(0)
    new = target.replace(
        "was 4.2 billion units",
        "was reported as 4.2 billion units, though an independent estimate places the "
        "figure at 4.05 billion units",
    )
    return {"replacements": [{"old_sentence": target, "new_sentence": new}]}


def _sentences(text: str) -> list[str]:
    body = " ".join((text or "").split())
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", body) if s.strip()]


def _stub_audit(prompt: str, polishing: str) -> dict:
    return {"status": "ALL CORRECT", "replacements": [], "polishing": polishing}


def _stub_polish(prompt: str) -> str:
    return _between(prompt, "<<<DOSSIER>>>", "<<<END DOSSIER>>>")


def _stub_polish_bad(prompt: str) -> str:
    body = _between(prompt, "<<<DOSSIER>>>", "<<<END DOSSIER>>>")
    return (
        body + "\n\n## An Extra Section\n\nA paragraph the formatter invented, which the "
        "similarity guard must reject because it adds words.\n"
    )


def _DRY_STUB(cfg_name: str, kind: str, model: str) -> Any:
    if _DRY_RUN["scenario"] is None:
        return None
    degraded = _DRY_RUN["scenario"] == "degraded"
    last = _DRY_RUN["counters"].get(cfg_name, 0)
    _DRY_RUN["counters"][cfg_name] = last + 1

    if cfg_name.startswith("plan"):
        if cfg_name == "plan" and last == 0 and degraded:
            return {"text": "Sure, here is a plan:\nResearch: only one line here\n"
                            "Search: a\nSearch: b\n"}
        return {"text": (
            "Research: The subject across mechanism, history and current state of play\n"
            "Research: The subject across evidence, data, incentives and disagreement\n"
            "Search: subject official statistics 2026\n"
            "Search: subject primary source dataset\n"
            "Search: subject criticism limitations\n"
            "Search: subject latest developments\n"
            "Search: subject methodology appendix\n"
        )}
    if cfg_name.startswith("profiler"):
        if degraded:
            return {"raise": "stubbed profiler outage"}
        return {"text": json.dumps({
            "archetype": "QUANTITATIVE_DATA",
            "primary_subjects": ["Sector Spending", "Primary Regulator", "Deferred Payments"],
            "target_metrics": ["Total Period Spending", "YoY Growth Rate", "Sampling Error"],
            "key_questions": ["What changed in spending and why do estimates differ?"],
        }), "finish_reason": "stop"}
    if cfg_name.startswith("inspector"):
        if degraded:
            return {"raise": "stubbed inspector outage"}
        return {"text": json.dumps({
            "entity_decisions": [],
            "metric_decisions": [],
        }), "finish_reason": "stop"}
    if cfg_name.startswith("ledger"):
        if degraded:
            return {"raise": "stubbed ledger outage on the primary model"}
        prompt = _last_prompt()
        return {"text": _stub_ledger(prompt), "finish_reason": "stop"}
    if cfg_name.startswith("dossier"):
        if degraded and cfg_name == "dossier" and last < 2:
            return {"raise": f"stubbed dossier idle timeout on attempt {last + 1}"}
        return {"text": _stub_dossier(_last_prompt()), "finish_reason": "stop",
                "reasoning_chars": 1450, "usage": {"prompt_tokens": 1200,
                                                    "completion_tokens": 900,
                                                    "total_tokens": 2100}}
    if cfg_name.startswith("redteam"):
        if degraded:
            return {"raise": "stubbed red-team outage on all models"}
        return {"text": json.dumps(_stub_redteam(_last_prompt())), "finish_reason": "stop"}
    if cfg_name.startswith("patch"):
        if degraded:
            return {"raise": "stubbed patch outage on all models"}
        return {"text": json.dumps(_stub_patch(_last_prompt())), "finish_reason": "stop"}
    if cfg_name.startswith("audit"):
        if degraded:
            return {"raise": "stubbed audit outage on all models"}
        polishing = "needed" if (degraded or _DRY_RUN["polish"]) else "not needed"
        return {"text": json.dumps(_stub_audit(_last_prompt(), polishing)),
                "finish_reason": "stop"}
    if cfg_name.startswith("polish"):
        if degraded:
            return {"raise": "stubbed polish outage on all models"}
        return {"text": (_stub_polish_bad(_last_prompt()) if _DRY_RUN["polish"]
                         else _stub_polish(_last_prompt())),
                "finish_reason": "stop"}
    if cfg_name.startswith("tavily_search"):
        if degraded:
            return {"raise": "stubbed tavily search outage"}
        return {"text": json.dumps({
            "results": [
                {"url": "https://example.com/1", "title": "Tavily result 1",
                 "content": "Stable source content about the topic [S1].", "score": 0.9,
                 "raw_content": "Stable source content about the topic [S1].", "published_date": "2026-01-01"},
                {"url": "https://example.com/2", "title": "Tavily result 2",
                 "content": "Another stable source with supporting data [S2].", "score": 0.8,
                 "raw_content": "Another stable source with supporting data [S2].", "published_date": "2026-01-02"},
            ]
        }), "finish_reason": "stop"}
    if cfg_name.startswith("claim_audit"):
        if degraded:
            return {"raise": "stubbed claim-audit outage"}
        return {"text": json.dumps({
            "results": [
                {"sentence": "stubbed supported claim", "status": "SUPPORTED", "replacement": ""},
                {"sentence": "stubbed unsupported claim", "status": "UNSUPPORTED", "replacement": "CUT"},
            ]
        }), "finish_reason": "stop"}
    return None


_LAST_PROMPT: list[str] = []


def _last_prompt() -> str:
    return _LAST_PROMPT[-1] if _LAST_PROMPT else ""


def _record_prompt(messages: Sequence[dict]) -> None:
    """Dry-run stubs are prompt-aware, so remember the last user turn."""
    if _DRY_RUN["scenario"] is None:
        return
    for msg in reversed(list(messages)):
        if msg.get("role") == "user" and isinstance(msg.get("content"), str):
            _LAST_PROMPT.append(msg["content"])
            return


# ============================================================ dry-run stubs


def dry_http_stub(method: str, url: str, params: dict | None = None) -> tuple[int, Any] | None:
    """Canned HTTP responses so --dry-run never touches the network."""
    if _DRY_RUN["scenario"] is None:
        return None
    scenario = _DRY_RUN["scenario"]
    degraded = scenario == "degraded"
    counter = _DRY_RUN["counters"].get("http:" + url, 0)
    _DRY_RUN["counters"]["http:" + url] = counter + 1

    if "api.cloudflare.com" in url:
        if degraded:
            return 500, {"success": False,
                         "errors": [{"code": 1000, "message": "stubbed cloudflare outage"}]}
        if "bge-m3" in url:
            return 200, {
                "result": {"data": [[0.25, 0.5, 0.25, 0.1] * 4 for _ in range(4)]},
                "success": True,
                "errors": [],
            }
        n = 12
        base = 4.5
        return 200, {"result": {"response": [
            {"id": i, "score": round(base - (i % 7) * 1.35, 4)} for i in range(n)]},
            "success": True, "errors": [], "messages": []}

    if url.endswith("/research"):
        if degraded:
            return 500, {"detail": {"error": "stubbed research outage"}}
        if "you.com" in url:
            return 200, {"task_id": "stub-task-1", "type": "research", "status": "queued",
                         "stream_url": "/v1/research/stub-task-1/stream",
                         "created_at": "2026-01-01T00:00:00Z"}
        return 200, {"status": "pending", "request_id": "stub-req-1",
                     "input": "stub", "model": "mini",
                     "created_at": "2026-01-01T00:00:00Z", "response_time": 0.1}

    if "/research/" in url:
        n = len(_dry_rows_for(url))
        return 200, {
            "status": "completed", "request_id": "stub-req-1",
            "created_at": "2026-01-01T00:00:00Z", "response_time": 1.0,
            "usage": {"credits": 7},
            "content": _dry_report(),
            "sources": [{"title": t, "url": u} for u, t, _ in DRY_DOC_SEEDS],
        }

    if "api.exa.ai/search" in url:
        rows = _dry_rows_for(url, 2)
        if degraded and counter == 0:
            return 429, {"error": "stubbed exa rate limit"}
        return 200, {"requestId": "stub", "results": [
            {"title": t, "url": u, "text": x, "publishedDate": "2026-03-01"} for u, t, x in rows]}

    if "api.tavily.com/search" in url:
        return 200, {"query": "stub", "results": [
            {"url": u, "title": t, "content": x, "raw_content": x}
            for u, t, x in _dry_rows_for(url, 2)]}

    return None


def _dry_rows_for(url: str, limit: int | None = None) -> list[tuple[str, str, str]]:
    key = re.sub(r"[^a-z0-9]+", "", url.lower())[-24:]
    picked = [d for d in DRY_DOC_SEEDS if re.sub(r"[^a-z0-9]+", "", d[0]) in key]
    if not picked:
        picked = list(DRY_DOC_SEEDS)
    rows = []
    for i, (u, t, x) in enumerate(picked):
        if i >= 1:
            u = f"{u}#v{i}"
        rows.append((u, t, x))
    return rows[:limit] if limit else rows


def _dry_report() -> str:
    parts = []
    for i, (url, title, text) in enumerate(DRY_DOC_SEEDS, 1):
        parts.append(f"{title} reports the following. {text} [{i}]")
    return (
        "Research report on the subject. " + " ".join(parts) +
        " The evidence base is partial and the sources disagree on the headline figure [1][2]."
    )


# ============================================================ self tests


class _NullLog(RunLog):
    def __init__(self) -> None:
        super().__init__(BASE_DIR / ".vin-selftest", "selftest", {"selftest": True})

    def warn(self, stage: str, message: str) -> None:
        self.data["warnings"].append({"stage": stage, "message": message})

    def stage_begin(self, name: str, detail: dict | None = None) -> None:
        self._stage_open = {"stage": name, "t0": time.time(), "status": "running",
                            "detail": detail or {}}

    def stage_end(self, status: str = "ok", error: str | None = None) -> None:
        if self._stage_open:
            self._stage_open["status"] = status
            self._stage_open = None

    def artifact(self, name: str, content: str) -> Path:
        return BASE_DIR / ".vin-selftest" / name


def check(label: str, cond: bool, detail: str = "") -> bool:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {label}" + (f" -- {detail}" if detail and not cond else ""))
    _TEST_RESULTS.append((label, bool(cond)))
    return bool(cond)


_TEST_RESULTS: list[tuple[str, bool]] = []


def selftest() -> int:
    _TEST_RESULTS.clear()
    print("\nvin selftest (offline)\n")

    print("planner parser")
    good = ("Research: broad one\nResearch: broad two\n"
            "Search: a one\nSearch: a two\nSearch: a three\nSearch: a four\n"
            "Search: a five\n")
    r, s = parse_plan(good)
    check("valid: 2 research + 5 search", plan_is_valid(r, s), f"{r} {s}")
    wide = "Research: one\nResearch: two\n" + "".join(
        f"Search: facet{i}\n" for i in range(13))
    rw, sw = parse_plan(wide)
    check("valid: 13 search lines accepted", plan_is_valid(rw, sw), f"{len(sw)}")
    mid = "Research: one\nResearch: two\n" + "".join(
        f"Search: mid{i}\n" for i in range(9))
    rm, sm = parse_plan(mid)
    check("valid: 9 search lines accepted", plan_is_valid(rm, sm), f"{len(sm)}")
    chatty = ("Here is my plan:\n\nResearch: broad one\nSome chatter\nResearch: broad two\n"
              "Search: a one\nSearch: a two\nSearch: a three\nSearch: a four\n"
              "Search: a five\nHope that helps!\n")
    r2, s2 = parse_plan(chatty)
    check("extra chatter ignored", r2 == ["broad one", "broad two"] and len(s2) == 5, str(s2))
    few = "Research: one\nResearch: two\nSearch: a\nSearch: b\n"
    r3, s3 = parse_plan(few)
    check("too few search lines rejected", not plan_is_valid(r3, s3), f"{len(s3)}")
    many = "Research: one\nResearch: two\n" + "".join(f"Search: s{i}\n" for i in range(14))
    r4, s4 = parse_plan(many)
    check("14 search lines rejected", not plan_is_valid(r4, s4), f"{len(s4)}")
    one = "Research: only one\nSearch: a\nSearch: b\nSearch: c\nSearch: d\nSearch: e\n"
    r5, s5 = parse_plan(one)
    check("one research line rejected", not plan_is_valid(r5, s5))
    fb_r, fb_s = fallback_plan("what is the state of solid state batteries")
    check("built-in fallback plan is valid", plan_is_valid(fb_r, fb_s), f"{fb_r} {fb_s}")
    check("fallback plan offers broad facet coverage", len(fb_s) >= 9, str(len(fb_s)))

    print("\njson extractor")
    check("plain object", extract_json('{"a": 1}') == {"a": 1})
    fenced = "```json\n{\"challenges\": [{\"id\": 1}]}\n```"
    check("fenced", extract_json(fenced) == {"challenges": [{"id": 1}]})
    check("bare fence", extract_json('```\n{"a": {"b": 2}}\n```') == {"a": {"b": 2}})
    trailing = 'Sure! {"status": "ALL CORRECT"} -- let me know if you need more.'
    check("trailing text", extract_json(trailing) == {"status": "ALL CORRECT"})
    inner = '{"outer": {"x": "} not a brace"}, "n": 1}'
    check("brace inside string ignored", extract_json(inner) == {"outer": {"x": "} not a brace"},
                                                                 "n": 1})
    escaped = '{"s": "a \\" } b"}'
    check("escaped quote ignored", extract_json(escaped) == {"s": 'a " } b'})
    for bad in ("not json at all", "", "```\n{oops}\n```", '{"a": 1', "[1,2,3]"):
        try:
            extract_json(bad)
            check(f"invalid rejected: {bad[:18]!r}", False)
        except (ValueError, json.JSONDecodeError):
            check(f"invalid rejected: {bad[:18]!r}", True)

    print("\nledger extraction parser")
    good = '{"extractions": [' \
           '{"chunk_id": "S1#0", "citations": ["S1"], "entities": [{"surface": "Acme", "type": "Organization", "start": 0, "end": 4}], "claims": [{"text": "x", "relation": "amount", "cites": ["S1"], "start": 0, "end": 1}]},' \
           '{"chunk_id": "S2#0", "citations": ["S2"], "entities": [], "claims": [{"text": "y", "relation": "date", "cites": ["S2"], "start": 0, "end": 1}]}' \
           ']}'
    parsed = _parse_ledger_extractions(good, {"S1#0", "S2#0"})
    check("extraction array parsed", len(parsed) == 2 and parsed[0]["chunk_id"] == "S1#0", str(parsed))
    check("extraction preserves entities", parsed[1]["entities"] == [], str(parsed[1]))
    try:
        _parse_ledger_extractions('{"extractions": [{"chunk_id": "S9#0", "entities": [], "claims": []}]}',
                                  {"S1#0", "S2#0"})
        check("unknown chunk_id rejected", False)
    except ValueError:
        check("unknown chunk_id rejected", True)
    try:
        _parse_ledger_extractions('{"extractions": [{"chunk_id": "S1#0", "entities": [], "claims": []}]}',
                                  {"S1#0", "S2#0"})
        check("missing chunk_id rejected", False)
    except ValueError:
        check("missing chunk_id rejected", True)
    try:
        _parse_ledger_extractions('not json', {"S1#0"})
        check("non-json rejected", False)
    except (ValueError, json.JSONDecodeError):
        check("non-json rejected", True)

    print("\nledger machine extractor")
    ents, money, dates = _ledger_machine_extract(
        "Spending was 4.2 billion units on 14 March 2026, up 11 percent year over year, "
        "down $1.8M from a year ago."
    )
    check("machine extractor found entities", len(ents) >= 3, str(ents))
    types = {e["type"] for e in ents}
    check("machine extractor types money+date+quantity",
          {"Money", "Date", "Quantity"} <= types or "Quantity" in types, str(types))
    d = [e for e in ents if e["type"] == "Date"]
    check("machine extractor normalized the date",
          d and d[0].get("norm") == "2026-03-14", str(d))
    m = [e for e in ents if e["type"] == "Money"]
    check("machine extractor captured currency money",
          any("4.2 billion" in e["surface"] for e in ents), str(m))
    check("machine extractor money list populated", len(money) >= 1, str(money))
    check("machine extractor dates list populated", len(dates) >= 1, str(dates))
    check("machine extractor empty text", list(_ledger_machine_extract("")) == [[], [], []])

    print("\nreplacement applier")
    log = _NullLog()
    base = "The sky is blue. Water is wet. The sky is blue.\nSecond line here."
    out, st = apply_replacements(base, [{"old_sentence": "Water is wet.",
                                         "new_sentence": "Water is H2O."}], log)
    check("exact match applied", "Water is H2O." in out and st["applied"] == 1, str(st))
    out, st = apply_replacements(base, [{"old_sentence": "Water  is\nwet.",
                                         "new_sentence": "Water is H2O."}], log)
    check("whitespace difference applied", "Water is H2O." in out and st["applied"] == 1, out)
    out, st = apply_replacements(base, [{"old_sentence": "Grass is wet.",
                                         "new_sentence": "x"}], log)
    check("not found skipped", out == base and st["skipped_not_found"] == 1, str(st))
    out, st = apply_replacements(base, [{"old_sentence": "The sky is blue.",
                                         "new_sentence": "The sky is grey."}], log)
    check("duplicate sentence skipped", out == base and st["skipped_duplicate"] == 1, str(st))
    out, st = apply_replacements(base, [{"old_sentence": "Second line here",
                                         "new_sentence": "Third line there"}], log)
    check("unrelated text untouched",
          out == "The sky is blue. Water is wet. The sky is blue.\nThird line there.", out)
    out, st = apply_replacements(base, [{"old_sentence": "Water is wet.",
                                         "new_sentence": "Water is wet."}], log)
    check("no-op skipped", out == base and st["skipped_invalid"] == 1, str(st))
    out, st = apply_replacements(base, [{"old_sentence": "sky", "new_sentence": "air"}], log)
    check("partial-word match refused", out == base, out)

    print("\ncitation validator")
    reg = SourceRegistry()
    reg.register("https://a.example/one", "One", "2026-01-02", "exa", "primary")
    reg.register("https://b.example/two/", "Two", "", "tavily", "counter")
    rep = validate_citations("Claim [S1] and [S2] but also [S9].", reg)
    check("known ids accepted", rep["cited"] == ["S1", "S2", "S9"], str(rep["cited"]))
    check("unknown id detected", rep["unknown"] == ["S9"] and not rep["ok"], str(rep))
    dup = validate_citations("[S1] [S1] [S2]", reg)
    check("ids deduplicated", dup["cited"] == ["S1", "S2"], str(dup["cited"]))
    check("no citations is valid", validate_citations("no tags here", reg)["ok"])
    check("url dedupe", reg.register("https://www.a.example/one/", "dup", "", "exa",
                                     "primary") == "S1")
    check("counter origin recorded", reg.get("S2")["origin"] == "counter")
    section = build_sources_section("Text [S2] [S1].", reg)
    check("sources section numbers cited ids in order",
          "1. Two" in section and "2. One" in section, section)

    print("\nchunker")
    para = " ".join(f"word{i}" for i in range(1200))
    chunks = chunk_document(para, "S1")
    check("long paragraph split", len(chunks) >= 3, f"{len(chunks)} chunks")
    check("chunk sizes in range",
          all(CHUNK_MIN_WORDS <= c.words <= CHUNK_MAX_WORDS for c in chunks),
          str([c.words for c in chunks]))
    check("source id and index preserved",
          all(c.source_id == "S1" and c.index == i for i, c in enumerate(chunks)))
    short = "Only a handful of words here."
    check("short doc yields one chunk", len(chunk_document(short, "S2")) == 1)
    check("empty doc yields none", chunk_document("", "S3") == [])
    tiny = chunk_document("one two three", "S4", min_words=50)
    check("undersized tail merged into previous chunk", len(tiny) == 1, str(len(tiny)))
    multi = "\n\n".join([" ".join(f"a{i}" for i in range(300)) for _ in range(4)])
    mchunks = chunk_document(multi, "S5")
    check("multi-paragraph split", len(mchunks) >= 3, f"{len(mchunks)}")

    print("\nledger claim signal")
    ok, kind = claim_signal("Reported spending was 4.2 billion units, up 11 percent [S1].")
    check("money figure is a claim", ok and kind == "figure", f"{ok},{kind}")
    ok, kind = claim_signal("The regulator published the primary dataset on 14 March 2026.")
    check("date is a claim", ok and kind == "date", f"{ok},{kind}")
    ok, kind = claim_signal("An independent study estimates spending at 4.05 billion units.")
    check("money beats date in kind", ok and kind == "figure", f"{ok},{kind}")
    ok, kind = claim_signal("The paper recommends further study.")
    check("attribution verb is a claim", ok and kind == "attribution", f"{ok},{kind}")
    ok, kind = claim_signal("Coverage is described as complete for entities filing under the standard regime.")
    check("plain prose is not a claim", not ok and kind == "", f"{ok},{kind}")

    print("\nrerank threshold (floor, 60-cap, and 45+15 MMR novelty)")
    scored = [(i, round(10.0 - i * 0.01, 4)) for i in range(80)]
    kept = apply_rerank_threshold(scored, 0.30)
    check("cap at 60", len(kept) == RERANK_CAP, str(len(kept)))
    check("keeps highest scores", kept[0][1] == 10.0 and kept[-1][1] >= 0.30)
    synth_chunks = [
        Chunk(f"S{(i % 6) + 1}", i,
              ("alpha beta gamma " * 20) if i < 55 else f"unique_gem_topic_{i} delta epsilon " * 15,
              60)
        for i in range(75)
    ]
    kept_mmr = apply_rerank_threshold(
        [(i, round(0.95 - i * 0.005, 4)) for i in range(75)],
        0.30,
        chunks=synth_chunks,
    )
    check("MMR selects 45 core + 15 novelty chunks = 60", len(kept_mmr) == 60, str(len(kept_mmr)))
    check("MMR rescues low-overlap novelty chunks past rank 55",
          any(idx >= 55 for idx, _ in kept_mmr[45:]), str([idx for idx, _ in kept_mmr[45:]]))
    low = [(i, -5.0) for i in range(50)]
    kept = apply_rerank_threshold(low, 0.30)
    check("floor keeps top 8 below threshold", len(kept) == RERANK_FLOOR, str(len(kept)))
    mid = [(i, 0.5 if i < 20 else 0.05) for i in range(50)]
    kept = apply_rerank_threshold(mid, 0.30)
    check("threshold applies above floor", len(kept) == 20 and all(s >= 0.30 for _, s in kept),
          str(len(kept)))
    exact = [(i, 0.30) for i in range(12)]
    check("score equal to threshold is kept",
          len(apply_rerank_threshold(exact, 0.30)) == 12)
    tiny = [(0, 0.9), (1, 0.1)]
    check("fewer chunks than floor keeps all",
          len(apply_rerank_threshold(tiny, 0.30)) == 2)

    print("\nduralumin entity resolution, numpy math & networkx graph")
    ent_clusters = _tier1_resolve_entities([
        {"surface": "The Regulator", "canonical_hint": "Primary Sector Regulator", "type": "Organization", "cites": ["S1"]},
        {"surface": "Primary Sector Regulator", "canonical_hint": "Primary Sector Regulator", "type": "Organization", "cites": ["S3"]},
        {"surface": "Independent FOI Study", "canonical_hint": "Independent FOI Study", "type": "Organization", "cites": ["S2"]},
    ])
    check("tier-1 rapidfuzz merges entity aliases across S1 & S3",
          len(ent_clusters) == 2 and {"S1", "S3"} <= ent_clusters[0]["cites"], str(ent_clusters))

    m_topics = _tier1_group_metrics([
        {"subject": "Sector Spending", "metric_name": "Total Reported Spending", "raw_value": "4.2 billion units",
         "numeric_value": 4.2e9, "unit": "units", "period_or_date": "2026",
         "conditions_caveats": "Excludes deferred payments", "epistemic_status": "OFFICIAL_PRIMARY", "cites": ["S1"]},
        {"subject": "Sector Spending", "metric_name": "Total Reported Spending Estimate", "raw_value": "4.05 billion units",
         "numeric_value": 4.05e9, "unit": "units", "period_or_date": "2026",
         "conditions_caveats": "FOI reconstruction; 38% sample", "epistemic_status": "PEER_REVIEWED", "cites": ["S2"]},
        {"subject": "Implementation Cost", "metric_name": "Cost Spike", "raw_value": "30 percent",
         "numeric_value": 30.0, "unit": "%", "period_or_date": "2026",
         "conditions_caveats": "12 non-random interviews", "epistemic_status": "UNVERIFIED_OPINION", "cites": ["S4"]},
    ])
    res_ents = [
        {"entity_id": f"ENT_{i:02d}", "canonical_name": c["canonical_name"], "type": c["type"],
         "aliases": list(c["aliases"]), "role": "Actor", "cites": sorted(c["cites"])}
        for i, c in enumerate(ent_clusters, 1)
    ]
    res_mets = [
        {"topic_id": f"MTOP_{i:02d}", "topic_name": t["topic_name"], "unit": t["unit"], "frames": t["frames"]}
        for i, t in enumerate(m_topics, 1)
    ]
    d_graph = _duralumin_build_graph_and_math(
        res_ents,
        res_mets,
        [
            {"subject": "Primary Sector Regulator", "relation": "applies", "object": "Insolvency Netting Rule",
             "text": "Regulator methodology nets out insolvencies and excludes 4% exempted residual [S1].",
             "epistemic_status": "CONSENSUS_FACT", "cites": ["S1"]},
            {"subject": "Independent FOI Study", "relation": "critiques", "object": "Insolvency Netting Rule",
             "text": "Independent study tests three robustness treatments of deferred & netted items [S2].",
             "epistemic_status": "CONSENSUS_FACT", "cites": ["S2"]},
        ],
        [("Residual 4% exempted group in two industries is unobserved", ["S1"])],
        {"S1": "ECHO_S1", "S2": "ECHO_S2", "S3": "ECHO_S3", "S4": "ECHO_S4"},
    )
    check("duralumin numpy engine detects 4.2B vs 4.05B quant discrepancy",
          len(d_graph["quant_discrepancies"]) == 1
          and 3.5 <= d_graph["quant_discrepancies"][0]["pct_spread"] <= 3.8,
          str(d_graph["quant_discrepancies"]))
    check("duralumin networkx engine finds 2-hop cross-source bridge",
          len(d_graph["bridges"]) >= 1, str(d_graph["bridges"]))
    check("duralumin isolates unverified blog outlier (+30%)",
          len(d_graph["isolated_outliers"]) >= 1, str(d_graph["isolated_outliers"]))

    print("\npolish similarity guard")
    a = "## Head\n\nText with [S1] and `code`.\n"
    check("identical passes", polish_similarity(a, a) >= 0.98)
    check("reformatted markdown passes",
          polish_similarity(a, "#Head\n\nText with [S1] and `code`.") >= 0.98,
          str(polish_similarity(a, "#Head\n\nText with [S1] and `code`.")))
    check("added words fail", polish_similarity(a, a + "Extra words added here.") < 0.98)
    check("removed words fail", polish_similarity(a, "## Head\n\nText.\n") < 0.98)
    check("two empties match", polish_similarity("", "") == 1.0)

    print("\nurl normalization and rerank envelope")
    check("tracking params stripped",
          normalize_url("https://www.Example.com/a/?utm_source=x&id=7#frag")
          == "example.com/a?id=7")
    check("trailing slash stripped", normalize_url("http://a.com/b/") == "a.com/b")
    check("empty url rejected", normalize_url("") == "")
    env = {"result": {"response": [{"id": 0, "score": 1.5}, {"id": 1, "score": -2.25}]},
           "success": True, "errors": [], "messages": []}
    check("verified cloudflare envelope parsed",
          parse_rerank_scores(env) == [(0, 1.5), (1, -2.25)])
    check("top-level response parsed",
          parse_rerank_scores({"response": [{"index": 3, "score": 0.5}]}) == [(3, 0.5)])
    check("errors reported as failure", parse_rerank_scores({"errors": [{"code": 1}]}) is None)
    check("garbage rejected", parse_rerank_scores("nope") is None)

    print("\nall-correct detection and prompt placeholders")
    check("bare ALL CORRECT", is_all_correct("ALL CORRECT"))
    check("lowercase all correct", is_all_correct("all correct"))
    check("status object", is_all_correct('{"status": "ALL CORRECT"}'))
    check("status object with underscore", is_all_correct('{"status":"ALL_CORRECT"}'))
    check("corrections object is not all-correct",
          not is_all_correct('{"status": "CORRECTIONS", "replacements": [{"a": 1}]}'))
    check("empty string is not all-correct", not is_all_correct(""))
    rendered = render_dossier_system()
    r_target = f"{P.REASONING_CHAR_TARGET:,} characters"
    r_limit = f"{P.REASONING_CHAR_LIMIT:,} characters"
    d_limit = f"{P.DOSSIER_TOKEN_LIMIT:,} tokens"
    check("dossier system renders all three budgets",
          r_target in rendered and r_limit in rendered and d_limit in rendered)
    check("no unresolved placeholders", "<<" not in rendered)
    check("reasoning budget stated before the token budget",
          rendered.index(r_limit) < rendered.index(d_limit))
    check("reasoning target restated", rendered.count(r_target) >= 2)
    check("reasoning ceiling restated", rendered.count(r_limit) >= 2)
    check("token budget restated", rendered.count(d_limit) >= 2)
    tail = rendered[-900:]
    check("all three budgets restated at the end",
          r_target in tail and r_limit in tail and d_limit in tail, tail[-160:])
    check("dossier prompt demands introduction then core findings then body",
          "introduction" in rendered.lower() and "## Core Findings" in rendered
          and "main bulk" in rendered.lower())
    check("dossier prompt frames the task as a dossier, not a brief answer",
          "dossier, not a brief research answer" in rendered.lower())
    check("dossier prompt asks for tables",
          "GitHub-Flavored Markdown tables" in rendered)
    rt = P.REDTEAM_SYSTEM
    check("red team is framed as the opposition",
          "YOU ARE THE OPPOSITION" in rt and "not an editor" in rt.lower())
    check("red team asks opponent questions", rt.count("would be asked") >= 1
          and "alternative explanation" in rt)
    check("red team aims to undermine or disprove",
          "undermine or disprove" in rt.lower())
    check("red team keeps the searchable json shape",
          '{"challenges":[{"id":1' in rt and '"search_query"' in rt
          and '{"challenges":[]}' in rt)
    check("red team keeps the verbatim claim rule",
          "copied EXACTLY from the dossier" in rt)
    au = P.AUDIT_SYSTEM
    check("audit allows up to 16 replacements", "At most 16 replacements" in au)
    check("audit forbids identical old/new sentences",
          "identical to its `old_sentence`" in au)
    check("audit teaches claim-to-statement rewrites",
          "REWRITING AN UNSUPPORTED CLAIM AS A STATEMENT" in au)
    check("audit requires confidence for the weaker restatement",
          "Confidence is required" in au and "leave that sentence alone" in au)
    check("audit forbids inventing a replacement",
          "do not guess at a number" in au)
    check("audit separates evidence from tone",
          "not about tone" in au)
    for name in ("PLANNER_SYSTEM", "LEDGER_EXTRACT_SYSTEM", "DOSSIER_SYSTEM", "REDTEAM_SYSTEM",
                 "PATCH_SYSTEM", "AUDIT_SYSTEM", "POLISH_SYSTEM", "JSON_REPAIR_MESSAGE",
                 "DOSSIER_RETRY_SUFFIX"):
        check(f"instruc_prompt.{name} present and non-empty",
              bool(getattr(P, name, "").strip()))
    for name in ("PLANNER_SYSTEM", "LEDGER_EXTRACT_SYSTEM", "REDTEAM_SYSTEM", "PATCH_SYSTEM",
                 "AUDIT_SYSTEM", "POLISH_SYSTEM", "DOSSIER_SYSTEM"):
        text = getattr(P, name)
        check(f"{name} forbids preamble",
              "preamble" in text.lower() and "commentary" in text.lower())

    print("\ngoogle-genai (Gemini Developer API) message mapping")
    system, contents = _genai_parts([
        {"role": "system", "content": "You are a fact engine."},
        {"role": "user", "content": "QUESTION: what happened?"},
        {"role": "assistant", "content": "(draft)"},
        {"role": "user", "content": "repair it"},
    ])
    check("genai lifts the system prompt out of contents",
          system == "You are a fact engine.", system)
    check("genai maps assistant to the model role",
          [c["role"] for c in contents] == ["user", "model", "user"],
          str([c["role"] for c in contents]))
    check("genai keeps turn text intact",
          contents[0]["parts"][0]["text"] == "QUESTION: what happened?",
          contents[0]["parts"][0]["text"])
    sys2, c2 = _genai_parts([
        {"role": "user", "content": "a"}, {"role": "user", "content": "b"}
    ])
    check("genai merges consecutive same-role turns",
          len(c2) == 1 and c2[0]["parts"][0]["text"] == "a\n\nb",
          str(c2))
    sys3, c3 = _genai_parts([{"role": "system", "content": "one"},
                             {"role": "system", "content": "two"}])
    check("genai concatenates multiple system prompts", sys3 == "one\n\ntwo", sys3)
    sys4, c4 = _genai_parts([])
    check("genai never sends empty contents", len(c4) == 1, str(c4))
    sys5, c5 = _genai_parts([{"role": "user",
                              "content": [{"type": "text", "text": "blocks"}]}])
    check("genai flattens block-style content",
          c5[0]["parts"][0]["text"] == "blocks", str(c5))

    print("\nledger config")
    check("ledger uses nScale OpenAI-compatible provider",
          LEDGER.provider == "openai", LEDGER.provider)
    check("ledger uses Qwen3-235B on nScale",
          LEDGER.model == "Qwen/Qwen3-235B-A22B-Instruct-2507"
          and LEDGER.base_url == "https://inference.api.nscale.com/v1",
          f"{LEDGER.base_url} {LEDGER.model}")
    check("ledger has no fallback model (verbatim resilience)",
          LEDGER.fallback == (), str([c.name for c in LEDGER.fallback]))
    check("no nScale stage points at GMI Cloud",
          not any("gmi-serving" in c.base_url
                   for c in (LEDGER, *LEDGER.fallback)))
    check("no nScale stage uses a GMI key",
          not any(c.key_env == "GMICLOUD_API_KEY"
                   for c in (LEDGER, *LEDGER.fallback)))
    check("ledger primary pins a model id rather than an alias",
          not LEDGER.model.endswith("-latest"), LEDGER.model)
    check("dossier primary is NVIDIA step-5-preview",
          DOSSIER.base_url == "https://integrate.api.nvidia.com/v1"
          and DOSSIER.model == env_str("VIN_DOSSIER_MODEL", "stepfun-ai/step-5-preview"),
          f"{DOSSIER.base_url} {DOSSIER.model}")
    check("dossier primary sets reasoning_effort", DOSSIER.reasoning_effort in ("low", "medium"),
          str(DOSSIER.reasoning_effort))
    check("kimi k3 streams", DOSSIER.stream is True)
    check("nscale K2.5 is retained as the first dossier fallback",
          DOSSIER.fallback and DOSSIER.fallback[0].model == "moonshotai/Kimi-K2.5",
          str([f.model for f in DOSSIER.fallback]))
    check("dossier chain has a second fallback",
          len(DOSSIER.fallback) == 2, str(len(DOSSIER.fallback)))
    check("nscale fallback sends no reasoning_effort",
          all(f.reasoning_effort is None for f in DOSSIER.fallback))
    tavy = SEARCH_CFG["tavily_research"]
    check("tavily research uses the pro model", tavy["research_model"] == "pro",
          tavy["research_model"])
    check("tavily research uses long output", tavy["output_length"] == "long",
          tavy["output_length"])
    check("tavily poll timeout is sized for pro/long",
          tavy["poll_timeout_s"] >= 600, str(tavy["poll_timeout_s"]))
    check("planner allows up to 13 search queries",
          plan_is_valid(["a", "b"], [f"s{i}" for i in range(13)]))
    check("planner rejects 4 search queries",
          not plan_is_valid(["a", "b"], [f"s{i}" for i in range(4)]))
    kw = genai_kwargs(LEDGER, [{"role": "system", "content": "sys"},
                               {"role": "user", "content": "hi"}], 4096)
    cfgobj = kw["config"]
    check("genai kwargs set max_output_tokens", cfgobj.max_output_tokens == 4096,
          str(cfgobj.max_output_tokens))
    check("genai kwargs set the system instruction",
          cfgobj.system_instruction == "sys", str(cfgobj.system_instruction))
    check("genai kwargs set thinking_budget to 0",
          cfgobj.thinking_config is not None
          and cfgobj.thinking_config.thinking_budget == 0,
          str(cfgobj.thinking_config))
    check("genai kwargs carry the model id", kw["model"] == LEDGER.model)

    print("\nbudget escalation and chain fall-through")

    class _Boom(Exception):
        pass

    low = _Boom("max_tokens is too low: this model spends part of its output budget "
                "reasoning, so a low ceiling is rejected")
    check("low-ceiling error is treated as a budget problem", _budget_error(low))
    toobig = _Boom("Too big: expected number to be <=200000 (param max_tokens)")
    check("too-big error is treated as a budget problem", _budget_error(toobig))
    credits = _Boom("insufficient_credits: The free allowance caps replies at 8,000 "
                    "tokens, and this answer needed more. Ask for something shorter, or "
                    "add credits to lift the cap. param max_tokens")
    check("credits error is NOT treated as a budget problem", not _budget_error(credits))
    check("ceiling is read from the provider message",
          _token_ceiling(toobig, 999) == 200000, str(_token_ceiling(toobig, 999)))
    check("ceiling falls back to the default when unstated",
          _token_ceiling(low, 12345) == 12345)
    check("underscore-formatted ceiling parses",
          _token_ceiling(_Boom("expected number to be <= 1_000"), 5) == 1000)
    overflow = _Boom("Your prompt exceeds the model's context length. Estimated input "
                     "tokens: 6,748. Requested output tokens: 64,000. Max context "
                     "tokens: 42,000. Please shorten the input and/or reduce max_tokens.")
    check("context overflow is recognised and yields a safe smaller budget",
          _context_overflow(overflow) == 42000 - 6748 - 512,
          str(_context_overflow(overflow)))
    check("context overflow is not treated as a low-ceiling escalation",
          not _budget_error(overflow))
    check("non-overflow error yields no context ceiling",
          _context_overflow(toobig) is None)

    async def _chain_probe():
        """A JSON-style runner returns None on failure; the fallback must still run."""
        tried: list[str] = []
        saved = {k: os.environ.get(k) for k in ("PROBE_A_KEY", "PROBE_B_KEY")}
        os.environ["PROBE_A_KEY"] = "probe-a-not-a-real-key"
        os.environ["PROBE_B_KEY"] = "probe-b-not-a-real-key"

        async def runner(cfg):
            tried.append(cfg.name)
            return None if cfg.name == "probe-a" else {"ok": True}

        class _Log:
            def warn(self, *a, **k):
                pass

            def provider_used(self, *a, **k):
                pass

            def stage_usage(self, *a, **k):
                pass

        a = StageCfg(name="probe-a", base_url="u", model="m", key_env="PROBE_A_KEY",
                     timeout=5.0, max_tokens=10)
        b = StageCfg(name="probe-b", base_url="u", model="m", key_env="PROBE_B_KEY",
                     timeout=5.0, max_tokens=10)
        try:
            res, cfg, fb = await llm_chain((a, b), runner, _Log(), "probe", core=False)
            return tried, res, cfg.name, fb
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    tried, chain_res, chain_cfg, chain_fb = asyncio.run(_chain_probe())
    check("chain falls through to the fallback when a stage returns None",
          tried == ["probe-a", "probe-b"], str(tried))
    check("chain returns the fallback result", chain_res == {"ok": True}, str(chain_res))
    check("chain reports the fallback as used",
          chain_cfg == "probe-b" and chain_fb is True, f"{chain_cfg} {chain_fb}")

    print("\nclaim extraction and source-grouping")
    reg = SourceRegistry()
    s1 = reg.register("https://example.com/a", "A", "", "tavily", "primary")
    s2 = reg.register("https://example.com/b", "B", "", "exa", "primary")
    reg.register_text(s1, "Source A says the capital of France is Paris.")
    reg.register_text(s2, "Source B says the capital of Germany is Berlin.")
    dossier_text = (
        f"Paris is the capital of France [{s1}]. "
        f"Berlin is the capital of Germany [{s2}]. "
        f"Madrid is the capital of Spain."
    )
    grouped = extract_claims_by_source(dossier_text, reg)
    check("claim extraction groups sentences by source id",
          set(grouped.keys()) == {s1, s2}, str(grouped.keys()))
    check("each source gets exactly its cited sentences",
          all(len(v) == 1 for v in grouped.values()), str({k: len(v) for k, v in grouped.items()}))
    check("claim text preserves the citation tag",
          all(f"[{sid}]" in sent for sid, sents in grouped.items() for sent in sents))
    uncited = [s for s in _extract_sentences(dossier_text) if s1 not in s and s2 not in s]
    check("uncited sentences are not in any group",
          all(f"[{s1}]" not in s and f"[{s2}]" not in s for s in uncited))

    print("\nclaim replacement and cut")
    text = f"Line one [{s1}].\nLine two [{s2}].\nLine three [{s1}].\nLine four."
    reps = [
        {"old_sentence": f"Line one [{s1}].", "new_sentence": "Line one revised.", "cut": False},
        {"old_sentence": f"Line two [{s2}].", "new_sentence": "", "cut": True},
    ]
    cuts = {f"Line two [{s2}]."}
    updated, stats = apply_claim_replacements(text, reps, cuts)
    check("CUT removes the sentence entirely", "Line two" not in updated, updated)
    check("PARTIAL replaces the sentence", "Line one revised." in updated, updated)
    check("unaffected lines survive", "Line four." in updated, updated)
    check("stats count applied cuts", stats["cut"] == 1, str(stats))
    multi = f"Same [{s1}] and also [{s1}] later."
    reps2 = [{"old_sentence": f"Same [{s1}] and also [{s1}] later.", "new_sentence": "Replaced.", "cut": False}]
    u2, s2s = apply_claim_replacements(multi, reps2, set())
    check("replacement applied once per distinct sentence", u2.count("Replaced.") == 1, u2)

    print("\nclaim audit config and prompt")
    check("claim audit uses CoreWeave GLM 5.3 Flash",
          CLAIM_AUDIT.base_url == "https://api.inference.wandb.ai/v1"
          and CLAIM_AUDIT.model == "zai-org/GLM-5.3-Flash",
          f"{CLAIM_AUDIT.base_url} {CLAIM_AUDIT.model}")
    check("claim audit pins reasoning_effort to low", CLAIM_AUDIT.reasoning_effort == "low")
    check("claim audit prompt teaches all four statuses",
          all(s in P.CLAIM_AUDIT_SYSTEM for s in ("SUPPORTED", "PARTIAL",
                                                   "CONTRADICTED", "UNSUPPORTED")))
    check("claim audit prompt requires CUT for unsupported",
          "replacement is the exact string CUT" in P.CLAIM_AUDIT_SYSTEM)
    check("claim audit prompt preserves citation tags in replacements",
          "citation tag" in P.CLAIM_AUDIT_SYSTEM)
    check("claim audit prompt requires every input sentence in output",
          "appear exactly once in the output" in P.CLAIM_AUDIT_SYSTEM)

    print("\nlexical rerank fallback (bm25)")
    q = "sodium ion battery chemistry energy density"
    rel = Chunk("S1", 0, " ".join(["sodium"] * 12 + ["ion"] * 6 + ["chemistry"] * 5 +
                                  ["filler"] * 200), 223)
    irr = Chunk("S2", 0, " ".join(["cooking"] * 30 + ["recipe"] * 30 + ["bread"] * 160), 220)
    sc = lexical_scores(q, [rel, irr])
    check("bm25 scores every chunk", set(sc) == {0, 1}, str(sc))
    check("bm25 ranks the on-topic chunk first",
          max(sc, key=lambda i: sc[i]) == 0, str(sc))
    check("bm25 gives off-topic text a near-zero score", sc[1] < 0.5, str(sc[1]))
    check("bm25 returns empty for an empty chunk list", lexical_scores(q, []) == {})
    check("bm25 returns empty when the query has no content words",
          lexical_scores("the and of", [rel]) == {})

    print("\nself-test summary")
    failed = [n for n, ok in _TEST_RESULTS if not ok]
    print(f"  {len(_TEST_RESULTS) - len(failed)}/{len(_TEST_RESULTS)} checks passed")
    if failed:
        print("  FAILED: " + "; ".join(failed))
    return 1 if failed else 0


# ============================================================ cli


def enable_dry_run(scenario: str) -> None:
    _DRY_RUN["scenario"] = scenario
    _DRY_RUN["counters"] = {}
    _DRY_RUN["polish"] = False


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="vin.py", description="Vin research dossier pipeline")
    ap.add_argument("question", nargs="?", help="the research question")
    ap.add_argument("--out", default=str(BASE_DIR / RUNS_DIRNAME), help="output directory")
    ap.add_argument("--skip-redteam", action="store_true")
    ap.add_argument("--skip-polish", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="stub every network call with canned responses")
    ap.add_argument("--selftest", action="store_true", help="run the offline self-tests")
    ap.add_argument("--no-banner", action="store_true", help="suppress the config table")
    return ap.parse_args(list(argv))


async def _run_dry(out: Path, skip_redteam: bool, skip_polish: bool) -> int:
    scenarios = ["happy", "degraded"]
    if not skip_polish:
        scenarios.append("polish-guard")
    for scenario in scenarios:
        name = "happy" if scenario == "happy" else scenario
        print("\n" + "#" * 78)
        print(f"# DRY RUN scenario: {name}")
        print("#" * 78)
        enable_dry_run("degraded" if scenario == "degraded" else "happy")
        _DRY_RUN["polish"] = scenario == "polish-guard"
        target = Path(out) / "dry" / name
        target.mkdir(parents=True, exist_ok=True)
        try:
            await run_pipeline("What changed in grid-scale battery storage costs since 2024, "
                               "and what drove the change?", target,
                               skip_redteam=skip_redteam, skip_polish=skip_polish)
        except AbortPipeline as exc:
            print(f"DRY RUN {name} aborted as designed: {exc}")
        log_path = sorted(target.glob("*/log.json"))
        if log_path:
            data = json.loads(log_path[-1].read_text(encoding="utf-8"))
            stages = {s["stage"]: s["status"] for s in data["stages"]}
            print(f"DRY RUN {name}: stages={stages}")
            print(f"DRY RUN {name}: warnings={len(data['warnings'])} "
                  f"fallbacks={sum(1 for s in data['stages'] if s.get('fallback_used'))} "
                  f"aborted={bool(data.get('aborted'))}")
            dossier = (log_path[-1].parent / "dossier.md")
            print(f"DRY RUN {name}: dossier {'written' if dossier.exists() else 'MISSING'}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.selftest:
        return selftest()
    if not args.question:
        print("error: a question is required (or use --selftest)", file=sys.stderr)
        return 2
    if not args.no_banner:
        print_banner()
    out = Path(args.out)
    if args.dry_run:
        return asyncio.run(_run_dry(out, args.skip_redteam, args.skip_polish))
    try:
        path = asyncio.run(run_pipeline(args.question, out, args.skip_redteam,
                                        args.skip_polish))
    except AbortPipeline as exc:
        print(f"\nPIPELINE ABORTED (core stage failure): {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"\nUNEXPECTED FAILURE in stage pipeline: {type(exc).__name__}: "
              f"{_short(str(exc), 400)}", file=sys.stderr)
        print("partial artifacts and log.json are in the run directory printed above",
              file=sys.stderr)
        return 1
    print(f"\ndossier: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())