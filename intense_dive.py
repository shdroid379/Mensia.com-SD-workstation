import os
import re
import asyncio
import urllib.parse
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path, override=False)

import httpx
from exa_py import AsyncExa
from tavily import AsyncTavilyClient
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from mistralai.client import Mistral

from ai_instructions import (
    mistral_synthesis_instructions,
    mistral_synthesis_prompt,
    audit_synthesis_instructions,
    audit_synthesis_prompt
)

exakey = AsyncExa(api_key=os.getenv("EXA_API_KEY"))
tavilykey = AsyncTavilyClient(api_key=os.getenv("TAVILY_KEY"))

# Debug: verify remaining primary keys are loaded

# =====================================================================
# 1. SCRAPERS (No Individual UI Updates)
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

async def fetch_linkup_dossier(query: str, client: httpx.AsyncClient):
    headers = {"Authorization": f"Bearer {os.getenv('LINKUP_KEY')}"}

    init_res = await client.post(
        "https://api.linkup.so/v1/research",
        headers=headers,
        json={
            "q": query,
            "mode": "investigate",
            "reasoning_depth": "M",
            "outputType": "sourcedAnswer"
        }
    )
    task_id = init_res.json().get('id')
    if not task_id:
        return ("Third Plug", "No dossier available from this source.", [])

    attempts = 0
    max_attempts = 180

    while attempts < max_attempts:
        poll_res = await client.get(f"https://api.linkup.so/v1/research/{task_id}", headers=headers)
        job = poll_res.json()
                
        if job.get("status") == "completed":
            dossier_text = job.get("answer", job.get("output", "No text generated."))
            summary = f"THIRD PLUG AUTONOMOUS DOSSIER:\n{dossier_text}\n\n"
            sources = []
            for src in job.get("sources", []):
                sources.append({
                    "url": src.get('url', ''),
                    "content": src.get('snippet', src.get('content', ''))
                })
            return ("Third Plug", summary, sources)
            
        elif job.get("status") == "failed":
            return ("Third Plug", f"THIRD PLUG FAILED: {job.get('error')}\n\n", [])
            
        attempts += 1
        await asyncio.sleep(5)
        
    return ("Third Plug", "THIRD PLUG TIMEOUT: Agent exceeded maximum run window.\n\n", [])

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
                    
                    # Numbered citation header so models map claims to [idx]
                    master_dossier += f"[{idx}] Source ({url}):\n{content[:5000]}\n\n"
                    idx += 1

    return master_dossier, sources

async def fetch_combined_dossier_with_academic_papers(query: str, status_cb=None):
    if status_cb:
        status_cb("PULLING SOURCES ACROSS 4 DIFFERENT PLUGS...")
    async with httpx.AsyncClient(timeout=60.0) as http_client:
        results = await asyncio.gather(
            fetch_tavily_dossier(query),
            fetch_exa_dossier(query),
            fetch_linkup_dossier(query, http_client),
            semantic_scholar_data(query, http_client)
        )
        return compile_and_deduplicate(results, status_cb)

async def fetch_combined_dossier(query: str, status_cb=None):
    if status_cb:
        status_cb("PULLING SOURCES ACROSS 3 DIFFERENT PLUGS...")
    async with httpx.AsyncClient(timeout=60.0) as http_client:
        results = await asyncio.gather(
            fetch_tavily_dossier(query),
            fetch_exa_dossier(query),
            fetch_linkup_dossier(query, http_client)
        )
        return compile_and_deduplicate(results, status_cb)

# =====================================================================
# 3. SYNTHESIS & CASCADING AUDIT
# =====================================================================

async def synthesize_with_mistral(query: str, master_dossier: str, status_cb=None) -> str:
    if status_cb:
        status_cb("SYNTHESIZING THE SYNTHESIS...")

    try:
        mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
        formatted_user_prompt = mistral_synthesis_prompt.format(
            query=query, 
            master_dossier=master_dossier
        )

        response = await mistral_client.chat.complete_async(
            model="mistral-small-latest",
            messages=[
                {"role": "system", "content": mistral_synthesis_instructions},
                {"role": "user", "content": formatted_user_prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Mistral synthesis error: {e}")
        return master_dossier

async def audit_the_synthesis(query: str, draft_dossier: str, status_cb=None) -> str:
    if status_cb:
        status_cb("AUDITING THE DOSSIER...")

    formatted_user_prompt = audit_synthesis_prompt.format(
        query=query,
        draft_dossier=draft_dossier
    )

    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": audit_synthesis_instructions},
        {"role": "user", "content": formatted_user_prompt}
    ] 

    # 1. Primary Endpoint: CometAPI (Kimi K2)
    try:
        client_comet = AsyncOpenAI(
            api_key=os.getenv("COMET_API_KEY"),
            base_url="https://api.cometapi.com/v1" 
        )
        res = await client_comet.chat.completions.create(
            model="kimi-k2-250905", 
            messages=messages,
            temperature=0.1,
            max_tokens=8192 
        )
        
        raw_output = res.choices[0].message.content or ""
        
        # Strip out Kimi K2's internal <think> blocks
        cleaned_output = re.sub(r'<think>.*?</think>', '', raw_output, flags=re.DOTALL).strip()
        return cleaned_output

    except Exception as e:
        print(f"CometAPI (Kimi K2) failed: {e}")

    # 2. First Fallback: MorphLLM (GLM-5.3-Flash)
    try:
        client_morph = AsyncOpenAI(
            api_key=os.getenv("MORPH_API_KEY"),
            base_url="https://api.morphllm.com/v1" 
        )
        res = await client_morph.chat.completions.create(
            model="morph-glm53flash", 
            messages=messages,
            temperature=0.1,
            max_tokens=8192
        )
        raw_output = res.choices[0].message.content or ""
        return re.sub(r'<think>.*?</think>', '', raw_output, flags=re.DOTALL).strip()
    except Exception as e:
        print(f"MorphLLM (GLM-5.3-Flash) failed: {e}")

    # 3. Last Resort: FinalRouter (DeepSeek V4 Flash)
    try:
        client_final = AsyncOpenAI(
            api_key=os.getenv("FINALROUTER_API_KEY"),
            base_url="https://finalrouter.com/api/v1" 
        )
        res = await client_final.chat.completions.create(
            model="deepseek-ai/DeepSeek-V4-Flash", 
            messages=messages,
            temperature=0.1,
            max_tokens=8192
        )
        raw_output = res.choices[0].message.content or ""
        return re.sub(r'<think>.*?</think>', '', raw_output, flags=re.DOTALL).strip()
    except Exception as e:
        print(f"FinalRouter (DeepSeek V4 Flash) failed: {e}")

    # 4. Absolute Failsafe: Return the Un-Audited Draft silently
    return draft_dossier

# Backward compatibility alias
audit_with_deepseek = audit_the_synthesis