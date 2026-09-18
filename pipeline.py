import os
import urllib.parse
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

from mcp_client import call_exa_async, call_tavily_async, call_you_async


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
    """Search mode: sequential fallback Exa -> Tavily -> You.com.
    Each call goes through its respective MCP server. Fully async.
    """
    # 1. Try Exa first
    try:
        result = await call_exa_async(prompt, count=5, mode="neural")
        if not result.get("error") and result.get("results"):
            items = [(r["url"], r.get("text", "")) for r in result["results"]]
            if items:
                return _format(items)
    except Exception as e:
        print(f"Exa MCP failed: {e}, trying Tavily...")

    # 2. Try Tavily
    try:
        result = await call_tavily_async(prompt, count=5, search_depth="basic")
        if not result.get("error") and result.get("results"):
            items = [(r["url"], r.get("content", "")) for r in result["results"]]
            if items:
                return _format(items)
    except Exception as e:
        print(f"Tavily MCP failed too: {e}, trying You.com...")

    # 3. Try You.com
    try:
        result = await call_you_async(prompt, count=5)
        if not result.get("error") and result.get("results"):
            items = [(r["url"], r.get("content", "")) for r in result["results"]]
            if items:
                return _format(items)
    except Exception as e:
        print(f"You.com MCP failed too: {e}.")

    return "All search engines failed. Sorry for inconvenience, please try again later.", []
