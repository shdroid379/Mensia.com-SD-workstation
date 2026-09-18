import os
import re
import json
import asyncio
import urllib.parse
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path, override=False)

import httpx
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from google import genai

from ai_instructions import (
    intense_dive_synthesis_instructions,
    intense_dive_synthesis_prompt,
    intense_dive_auditor_instructions,
    intense_dive_auditor_prompt
)

from exa_py import AsyncExa
from tavily import AsyncTavilyClient

exakey = AsyncExa(api_key=os.getenv("EXA_API_KEY"))
tavilykey = AsyncTavilyClient(api_key=os.getenv("TAVILY_KEY"))

# =====================================================================
# 1. SCRAPERS (Direct async SDK calls - no MCP)
# =====================================================================

async def fetch_tavily_dossier(query: str):
    response = await tavilykey.search(
        query=query,
        max_results=8,
        search_depth="advanced",
        include_raw_content="markdown",
        include_answer="advanced"
    )
    summary = f"FIRST PRE-SYNTHESIZED PERSPECTIVE:\n{response.get('answer', 'No answer provided')}\n\n"
    sources = []
    for res in response.get("results", []):
        sources.append({
            "url": res.get('url', ''),
            "content": res.get('raw_content', res.get('content', ''))
        })
    return ("First Plug", summary, sources)

async def fetch_exa_dossier(query: str):
    response = await exakey.search(
        query=query,
        num_results=10,
        contents={"text": {"verbosity": "full"}, "subpages": 2},
        type="deep-reasoning"
    )
    sources = []
    for res in response.results:
        sources.append({"url": res.url, "content": res.text})
    return ("Second Plug", "", sources)


async def fetch_you_dossier(query: str, client: httpx.AsyncClient, status_cb=None):
    """You.com research via REST API (direct httpx call)."""
    api_key = os.getenv("YOU_API_KEY")
    if not api_key:
        return ("Third Plug", "THIRD PLUG FAILED: YOU_API_KEY not configured\n\n", [])

    try:
        init_res = await client.post(
            "https://api.you.com/v1/research",
            headers={
                "X-API-Key": api_key,
                "Content-Type": "application/json",
            },
            json={
                "input": query,
                "research_effort": "standard",
            },
        )
        init_res.raise_for_status()
        data = init_res.json()

        output = data.get("output") or {}
        dossier_text = output.get("content", "No text generated.")
        summary = f"THIRD PLUG AUTONOMOUS DOSSIER:\n{dossier_text}\n\n"

        sources = []
        for src in output.get("sources", []):
            sources.append({
                "url": src.get("url", ""),
                "content": src.get("snippets", [""])[0] if src.get("snippets") else src.get("title", ""),
            })

        return ("Third Plug", summary, sources)
    except Exception as e:
        print(f"You.com dossier failed: {e}")
        return ("Third Plug", f"THIRD PLUG FAILED: {e}\n\n", [])


async def semantic_scholar_data(query: str, client: httpx.AsyncClient):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": 5,
        "fields": "title,abstract,authors,year,tldr,citationCount,url"
    }
    res = await client.get(url, params=params)
    dossier = "ACADEMIC LITERATURE DATA:\n"
    sources = []

    if res.status_code == 200:
        for paper in res.json().get("data", []):
            authors = ", ".join([a["name"] for a in paper.get("authors", [])])
            tldr = paper.get("tldr", {}).get("text", "N/A")
            p_url = paper.get("url") or ""
            title = paper.get("title") or "Academic Paper"
            abstract = paper.get("abstract") or ""

            dossier += f"Title: {title} ({paper.get('year')})\n"
            dossier += f"Authors: {authors} | Citations: {paper.get('citationCount')}\n"
            dossier += f"TLDR: {tldr}\n"
            dossier += f"Abstract: {abstract}\n\n"

            if p_url:
                sources.append({
                    "url": p_url,
                    "content": f"Title: {title} ({paper.get('year')})\nAuthors: {authors}\nAbstract: {abstract}"
                })
    else:
        dossier += "Failed to fetch academic data\n"

    return ("Semantic Scholar", dossier, sources)

# =====================================================================
# 2. DEDUPLICATOR & INDEX COMPILER
# =====================================================================

def compile_and_deduplicate(results: list, status_cb=None):
    if status_cb:
        status_cb("DOSSIERING...")
    master_dossier = ""
    seen_urls = set()
    sources = []
    idx = 1

    for source_name, summary, raw_sources in results:
        if summary and summary.strip():
            master_dossier += f"=== {source_name.upper()} SYNTHESIS PERSPECTIVE ===\n{summary}\n\n"

        if raw_sources:
            master_dossier += f"--- {source_name.upper()} RAW SOURCE DATA ---\n"
            for src in raw_sources:
                url = src.get("url", "")
                content = src.get("content") or ""
                if not url or not content:
                    continue

                clean_url = url.split("://")[-1].replace("www.", "").rstrip("/")
                if clean_url not in seen_urls:
                    seen_urls.add(clean_url)
                    domain = urllib.parse.urlparse(url).netloc.replace("www.", "") or clean_url.split("/")[0]
                    sources.append({"id": idx, "url": url, "domain": domain})

                    master_dossier += f"[{idx}] Source ({url}):\n{content[:5000]}\n\n"
                    idx += 1

    return master_dossier, sources


def _replace_failed_plugs(results: list, plug_names: tuple[str, ...]) -> list:
    normalized_results = []
    for plug_name, result in zip(plug_names, results):
        if isinstance(result, Exception):
            print(f"{plug_name} failed: {result}")
            normalized_results.append((plug_name, f"{plug_name} unavailable: {result}\n\n", []))
        else:
            normalized_results.append(result)
    return normalized_results


async def fetch_combined_dossier_with_academic_papers(query: str, status_cb=None):
    if status_cb:
        status_cb("PULLING SOURCES ACROSS 4 DIFFERENT PLUGS...")
    async with httpx.AsyncClient(timeout=60.0) as http_client:
        results = await asyncio.gather(
            fetch_tavily_dossier(query),
            fetch_exa_dossier(query),
            fetch_you_dossier(query, http_client, status_cb),
            semantic_scholar_data(query, http_client),
            return_exceptions=True,
        )
        results = _replace_failed_plugs(results, ("First Plug", "Second Plug", "Third Plug", "Semantic Scholar"))
        return compile_and_deduplicate(results, status_cb)

async def fetch_combined_dossier(query: str, status_cb=None):
    if status_cb:
        status_cb("PULLING SOURCES ACROSS 3 DIFFERENT PLUGS...")
    async with httpx.AsyncClient(timeout=60.0) as http_client:
        results = await asyncio.gather(
            fetch_tavily_dossier(query),
            fetch_exa_dossier(query),
            fetch_you_dossier(query, http_client, status_cb),
            return_exceptions=True,
        )
        results = _replace_failed_plugs(results, ("First Plug", "Second Plug", "Third Plug"))
        return compile_and_deduplicate(results, status_cb)

# =====================================================================
# 3. SYNTHESIS & CASCADING AUDIT
# =====================================================================

async def synthesize_with_mistral(query: str, master_dossier: str, status_cb=None) -> str:
    if status_cb:
        status_cb("SYNTHESIZING THE SYNTHESIS...")

    formatted_user_prompt = intense_dive_synthesis_prompt.format(
        query=query,
        master_dossier=master_dossier
    )

    try:
        gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        response = await gemini_client.aio.models.generate_content(
            model="gemini-3.6-flash",
            contents=formatted_user_prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=intense_dive_synthesis_instructions,
                temperature=0.1,
                max_output_tokens=16384,
            )
        )
        return response.text
    except Exception as e:
        print(f"Gemini synthesis error: {e}")
        try:
            return await _complete_with_fallback([
                {"role": "system", "content": intense_dive_synthesis_instructions},
                {"role": "user", "content": formatted_user_prompt},
            ])
        except Exception as fallback_error:
            print(f"LLM synthesis fallback failed: {fallback_error}")
            return master_dossier

def _final_completion_text(response) -> str:
    """Return visible assistant text, rejecting incomplete reasoning-only replies."""
    if not response.choices:
        raise ValueError("provider returned no choices")

    raw_output = response.choices[0].message.content
    if not isinstance(raw_output, str):
        raise ValueError("provider returned reasoning but no final answer")

    cleaned_output = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()
    if not cleaned_output:
        raise ValueError("provider returned an empty final answer")
    return cleaned_output


FALLBACK_LLM_PROVIDERS = (
    ("OpenRouter (Nemotron Ultra 550B)", "OPENROUTER_API_KEY", "https://openrouter.ai/api/v1", "nvidia/nemotron-3-ultra-550b-a55b:free"),
    ("CometAPI (GLM-5.3-Flash)", "COMET_API_KEY", "https://api.cometapi.com/v1", "glm-5.3-flash"),
    ("MorphLLM (GLM-5.3-744B)", "MORPH_API_KEY", "https://api.morphllm.com/v1", "morph-glm53-744b"),
    ("FinalRouter (DeepSeek V4 Flash)", "FINAL_ROUTER", "https://finalrouter.com/api/v1", "deepseek/deepseek-v4-flash"),
    ("Requesty (Nemotron 3 Ultra)", "REQUESTY_API_KEY", "https://router.requesty.ai/v1", "nvidia/nemotron-3-ultra-550b-a55b"),
    ("Z.ai (GLM-5.3-Flash)", "ZAI_API_KEY", "https://api.z.ai/api/paas/v4", "glm-5.3-flash"),
)


async def _complete_with_fallback(messages: list[ChatCompletionMessageParam]) -> str:
    for provider_name, key_name, base_url, model in FALLBACK_LLM_PROVIDERS:
        api_key = os.getenv(key_name)
        if not api_key:
            print(f"{provider_name} skipped: {key_name} is not configured")
            continue

        try:
            client = AsyncOpenAI(api_key=api_key, base_url=base_url)
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
                max_tokens=8192,
            )
            return _final_completion_text(response)
        except Exception as e:
            print(f"{provider_name} failed: {e}")

    raise RuntimeError("all fallback LLM providers failed")


async def audit_the_synthesis(query: str, draft_dossier: str, status_cb=None) -> str:
    if status_cb:
        status_cb("AUDITING THE DOSSIER...")

    formatted_user_prompt = intense_dive_auditor_prompt.format(
        query=query,
        draft_dossier=draft_dossier
    )
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": intense_dive_auditor_instructions},
        {"role": "user", "content": formatted_user_prompt}
    ]

    try:
        return await _complete_with_fallback(messages)
    except Exception as e:
        print(f"LLM audit fallback failed: {e}")

    return draft_dossier

# Backward compatibility alias
audit_with_deepseek = audit_the_synthesis
