import asyncio
import os
import urllib.parse
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

from exa_py import AsyncExa
from tavily import AsyncTavilyClient
import httpx

exa_client = AsyncExa(api_key=os.getenv("EXA_API_KEY"))
tavily_client = AsyncTavilyClient(api_key=os.getenv("TAVILY_KEY"))


async def combined_research(prompt):
    """Deep research: Exa deep + Tavily advanced + You.com standard in parallel."""

    async def _exa_deep():
        try:
            response = await exa_client.search(
                prompt, type="deep", num_results=7,
                contents={"text": {"max_characters": 5000}}
            )
            return [(item.url, item.text or "") for item in response.results]
        except Exception as e:
            print(f"Exa deep failed: {e}")
            return []

    async def _tavily_deep():
        try:
            answer = await tavily_client.search(
                query=prompt, max_results=7,
                search_depth="advanced", chunks_per_source="auto"
            )
            return [(r['url'], r['content']) for r in answer.get("results", [])]
        except Exception as e:
            print(f"Tavily deep failed: {e}")
            return []

    async def _you_deep():
        api_key = os.getenv("YOU_API_KEY")
        if not api_key:
            return []
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                res = await client.post(
                    "https://ydc-index.io/v1/search",
                    headers={"X-API-Key": api_key, "Content-Type": "application/json"},
                    json={"query": prompt, "count": 6, "livecrawl": "web", "livecrawl_formats": ["markdown"]}
                )
                res.raise_for_status()
                data = res.json()
                items = []
                for r in data.get("results", {}).get("web", []):
                    content = "\n".join(r.get("snippets", [])) or r.get("description", "")
                    items.append((r.get("url", ""), content))
                return items
        except Exception as e:
            print(f"You.com deep failed: {e}")
            return []

    results = await asyncio.gather(
        _exa_deep(),
        _tavily_deep(),
        _you_deep(),
        return_exceptions=True,
    )

    combined_content = ""
    seen_urls = set()
    sources = []
    idx = 1

    for item in results:
        if isinstance(item, BaseException) or not item:
            continue
        for url, text in item:
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            domain = urllib.parse.urlparse(url).netloc.replace("www.", "")
            sources.append({"id": idx, "url": url, "domain": domain or url})
            combined_content += f"[{idx}] Source ({url}):\n{text}\n\n"
            idx += 1

    if not combined_content:
        return "All search engines failed, try again later..", []

    return combined_content, sources
