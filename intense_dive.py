import os
import asyncio
import httpx
from exa_py import AsyncExa
from tavily import AsyncTavilyClient
from ai_instructions import (
    MISTRAL_INTENSE_DIVE_SYNTHESIS_INSTRUCTION,
    MISTRAL_SYNTHESIS_USER_PROMPT,
    DEEPSEEK_INTENSE_DIVE_AUDIT_INSTRUCTION,
    DEEPSEEK_AUDIT_USER_PROMPT
)

exakey = AsyncExa(api_key=os.getenv("EXA_API_KEY"))
tavilykey = AsyncTavilyClient(api_key=os.getenv("TAVILY_KEY"))

# =====================================================================
# 1. SCRAPERS
# =====================================================================

async def fetch_tavily_dossier(query: str, status_cb=None):
    if status_cb:
        status_cb("PULLING SOURCES FROM FIRST PLUG...")
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

async def fetch_exa_dossier(query: str, status_cb=None):
    if status_cb:
        status_cb("PULLING SOURCES FROM SECOND PLUG...")
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

async def fetch_linkup_dossier(query: str, client: httpx.AsyncClient, status_cb=None):
    if status_cb:
        status_cb("PULLING SOURCES FROM THIRD PLUG...")
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

    # 15-minute ceiling (180 attempts * 5 seconds = 900s)
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
            
        if status_cb:
            status_cb("PULLING SOURCES FROM THIRD PLUG...")
            
        attempts += 1
        await asyncio.sleep(5)
        
    return ("Third Plug", "THIRD PLUG TIMEOUT: Agent exceeded maximum run window.\n\n", [])

async def semantic_scholar_data(query: str, client: httpx.AsyncClient, status_cb=None):
    if status_cb:
        status_cb("FETCHING ACADEMIC PAPERS...")
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": 5,
        "fields": "title,abstract,authors,year,tldr,citationCount,url"
    }
    res = await client.get(url, params=params)
    dossier = "ACADEMIC LITERATURE DATA:\n"
    if res.status_code == 200:
        for paper in res.json().get("data", []):
            authors = ", ".join([a["name"] for a in paper.get("authors", [])])
            tldr = paper.get("tldr", {}).get("text", "N/A")
            dossier += f"Title: {paper.get('title')} ({paper.get('year')})\n"
            dossier += f"Authors: {authors} | Citations: {paper.get('citationCount')}\n"
            dossier += f"TLDR: {tldr}\n"
            dossier += f"Abstract: {paper.get('abstract')}\n\n"
    else:
        dossier += "Failed to fetch academic data\n"

    return ("Semantic Scholar", dossier, [])

# =====================================================================
# 2. DEDUPLICATOR
# =====================================================================

def compile_and_deduplicate(results: list, status_cb=None) -> str:
    if status_cb:
        status_cb("DOSSIERING...")
    master_dossier = ""
    seen_urls = set()
    
    for source_name, summary, sources in results:
        if summary:
            master_dossier += summary
            
        if sources:
            master_dossier += f"--- {source_name.upper()} RAW SOURCE DATA ---\n"
            for src in sources:
                url = src.get("url", "")
                
                # Forces empty data into a string so it doesn't crash on None
                content = src.get("content") or "" 
                
                if not url or not content:
                    continue
                    
                clean_url = url.split("://")[-1].replace("www.", "").rstrip("/")
                if clean_url not in seen_urls:
                    seen_urls.add(clean_url)
                    master_dossier += f"Source ({url}):\n{content[:5000]}\n\n"

    return master_dossier

async def fetch_combined_dossier_with_academic_papers(query: str, status_cb=None):
    if status_cb:
        status_cb("DIVING IN...")
    async with httpx.AsyncClient(timeout=60.0) as http_client:
        results = await asyncio.gather(
            fetch_tavily_dossier(query, status_cb),
            fetch_exa_dossier(query, status_cb),
            fetch_linkup_dossier(query, http_client, status_cb),
            semantic_scholar_data(query, http_client, status_cb)
        )
        return compile_and_deduplicate(results, status_cb)

async def fetch_combined_dossier(query: str, status_cb=None):
    if status_cb:
        status_cb("DIVING IN...")
    async with httpx.AsyncClient(timeout=60.0) as http_client:
        results = await asyncio.gather(
            fetch_tavily_dossier(query, status_cb),
            fetch_exa_dossier(query, status_cb),
            fetch_linkup_dossier(query, http_client, status_cb)
        )
        return compile_and_deduplicate(results, status_cb)

# =====================================================================
# 3. SYNTHESIS & AUDIT
# =====================================================================

from mistralai.client import Mistral
mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

async def synthesize_with_mistral(query: str, master_dossier: str, status_cb=None) -> str:
    if status_cb:
        status_cb("SYNTHESIZING THE SYNTHESIS...")
    formatted_user_prompt = MISTRAL_SYNTHESIS_USER_PROMPT.format(
        query=query, 
        master_dossier=master_dossier
    )

    response = await mistral_client.chat.complete_async(
        model="mistral-small-latest",
        messages=[
            {"role": "system", "content": MISTRAL_INTENSE_DIVE_SYNTHESIS_INSTRUCTION},
            {"role": "user", "content": formatted_user_prompt}
        ]
    )
    return response.choices[0].message.content

from openai import AsyncOpenAI
hyperbolic_client = AsyncOpenAI(
    api_key=os.getenv("HYPERBOLIC_API_KEY"),
    base_url="https://api.hyperbolic.xyz/v1"
)

async def audit_with_deepseek(query: str, draft_dossier: str, status_cb=None) -> str:
    if status_cb:
        status_cb("AUDITING THE SYNTHESIS...")
    formatted_user_prompt = DEEPSEEK_AUDIT_USER_PROMPT.format(
        query=query,
        draft_dossier=draft_dossier
    )

    response = await hyperbolic_client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3",
        messages=[
            {"role": "system", "content": DEEPSEEK_INTENSE_DIVE_AUDIT_INSTRUCTION},
            {"role": "user", "content": formatted_user_prompt}
        ],
        temperature=0.1
    )
    if status_cb:
        status_cb("THE DOSSIER IS ALMOST READY...")
    return response.choices[0].message.content