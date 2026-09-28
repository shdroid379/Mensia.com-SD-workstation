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


def _format(items: list[tuple[str, str]]):
    combined = ""
    sources = []
    seen_urls = set()
    idx = 1
    for url, text in items:
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        domain = urllib.parse.urlparse(url).netloc.replace("www.", "")
        sources.append({"id": idx, "url": url, "domain": domain or url})
        combined += f"[{idx}] Source ({url}):\n{text}\n\n"
        idx += 1
    return combined, sources


async def basic_search(prompt: str):
    """Search mode: sequential fallback Exa -> Tavily -> You.com. Async."""
    # 1. Try Exa
    try:
        response = await exa_client.search(
            prompt, num_results=5,
            contents={"text": {"max_characters": 4500}},
            type="neural"
        )
        items = [(item.url, item.text or "") for item in response.results]
        if items:
            return _format(items)
    except Exception as e:
        print(f"Exa failed: {e}, trying Tavily...")

    # 2. Try Tavily
    try:
        answer = await tavily_client.search(
            query=prompt, max_results=5,
            search_depth="basic", chunks_per_source="auto"
        )
        items = [(r['url'], r['content']) for r in answer.get("results", [])]
        if items:
            return _format(items)
    except Exception as e:
        print(f"Tavily failed too: {e}, trying You.com...")

    # 3. Try You.com
    api_key = os.getenv("YOU_API_KEY")
    if api_key:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                res = await client.post(
                    "https://ydc-index.io/v1/search",
                    headers={"X-API-Key": api_key, "Content-Type": "application/json"},
                    json={"query": prompt, "count": 5, "livecrawl": "web", "livecrawl_formats": ["markdown"]}
                )
                res.raise_for_status()
                data = res.json()
                items = []
                for r in data.get("results", {}).get("web", []):
                    content = "\n".join(r.get("snippets", [])) or r.get("description", "")
                    items.append((r.get("url", ""), content))
                if items:
                    return _format(items)
        except Exception as e:
            print(f"You.com failed too: {e}")

    return "All search engines failed. Sorry for inconvenience, please try again later.", []
