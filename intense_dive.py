import asyncio
import httpx
from exa_py import AsyncExa
from tavily import AsyncTavilyClient
import os 
from ai_instructions import (
    MISTRAL_INTENSE_DIVE_SYNTHESIS_INSTRUCTION,
    MISTRAL_SYNTHESIS_USER_PROMPT,
    DEEPSEEK_INTENSE_DIVE_AUDIT_INSTRUCTION,
    DEEPSEEK_AUDIT_USER_PROMPT
)
exakey = AsyncExa(api_key=os.getenv("EXA_API_KEY"))
tavilykey = AsyncTavilyClient(api_key=os.getenv("TAVILY_KEY"))
async def fetch_tavily_dossier(query: str):
    response = await tavilykey.search(
        query=query,
        max_results=8,
        search_depth="advanced",
        include_raw_content="markdown",
        include_answer="advanced"
    )
    dossier = f"FIRST PRE-SYNTHESIZED PERSPECTIVE: \n{response.get('answer', 'No answer provided')}\n\n" 

    dossier += "TAVILY RAW SOURCE CONTEXT:\n"
    for res in response.get("results", []):
        raw_text = res.get('raw_content', res.get('content', ''))
        dossier += f"Source ({res['url']}):\n{raw_text[:5000]}\n\n"
    return dossier

async def fetch_exa_dossier(query: str):
    response = await exakey.search(
        query=query,
        num_results=10,
        contents={
            "text": {"verbosity": "full"},
            "subpages": 2
        },
        type="deep-reasoning"
    ) 
    source_dossier = "2nd list of sources:\n"
    for res in response.results:
        source_dossier += f"Source ({res.url}): \n{res.text[:5000]}\n\n"

    return source_dossier

async def fetch_linkup_dossier(query: str, client: httpx.AsyncClient):
    headers = {"Authorization": f"Bearer {os.getenv('LINKUP_KEY')}"}

    init_res = await client.post(
            "https://api.linkup.so/v1/research",
            json={
                "q": query,
                "mode": "investigate",
                "reasoning_depth": "M",
                "outputType": "sourcedAnswer"
            }
    )
    task_id = init_res.json().get('id')
    if not task_id:
        return "No dossier available from this source."

    while True:
        poll_res = await client.get(f"https://api.linkup.so/v1/research/{task_id}", headers=headers)
        job = poll_res.json()
                
        if job.get("status") == "completed":
            dossier_text = job.get("answer", job.get("output", "No text generated."))
                
            # 4. GRAB THE RAW WEB SOURCES
            sources = job.get("sources", [])
            
            # 5. COMPILE IT FOR MISTRAL
            compiled_report = f"Source 3 AUTONOMOUS DOSSIER:\n{dossier_text}\n\n"
            compiled_report += f"SOURCE 3 RAW SOURCE DATA:\n"
            
            for source in sources:
                compiled_report += f"- Title: {source.get('name')}\n"
                compiled_report += f"  URL: {source.get('url')}\n"
                compiled_report += f"  Context: {source.get('snippet', source.get('content', ''))[:2000]}\n\n"
                
            return compiled_report
        elif job.get("status") == "failed":
            return f"SOURCE THREE FAILED: \n\n"
            
        await asyncio.sleep(10)


async def semantic_scholar_data(query: str, client: httpx.AsyncClient):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": 5,
        "fields": "title,abstract,authors,year,tldr,citationCount"
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
        dossier += f"Failed to fetch academic data"

    return dossier


async def fetch_combined_dossier_with_academic_papers(query: str):
    async with httpx.AsyncClient(timeout=60.0) as http_client:

        results = await asyncio.gather(
            fetch_tavily_dossier(query),
            fetch_exa_dossier(query),
            fetch_linkup_dossier(query, http_client),
            semantic_scholar_data(query, http_client)
        )
        master_dossier = "\n".join(str(res) for res in results if res)

        return master_dossier


async def fetch_combined_dossier(query: str):
    async with httpx.AsyncClient(timeout=60.0) as http_client:

        results = await asyncio.gather(
            fetch_tavily_dossier(query),
            fetch_exa_dossier(query),
            fetch_linkup_dossier(query, http_client)
        )
        master_dossier = "\n".join(str(res) for res in results if res)

        return master_dossier



import os
from mistralai.client import Mistral

mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

async def synthesize_with_mistral(query: str, master_dossier: str) -> str:
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

async def audit_with_deepseek(query: str, draft_dossier: str) -> str:
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
    return response.choices[0].message.content
    