import asyncio
import urllib.parse
from mcp_client import call_exa_async, call_tavily_async, call_you_async


async def combined_research(prompt):
    """Deep research: run all three MCP servers in parallel.
    Exa deep mode, Tavily advanced, You.com standard.
    """

    async def _exa_deep():
        try:
            result = await call_exa_async(prompt, count=7, mode="deep")
            if result.get("error"):
                raise RuntimeError(result["error"])
            return [("exa", r["url"], r.get("text", "")) for r in result.get("results", [])]
        except Exception as e:
            print(f"Exa deep MCP failed: {e}")
            return []

    async def _tavily_deep():
        try:
            result = await call_tavily_async(prompt, count=7, search_depth="advanced")
            if result.get("error"):
                raise RuntimeError(result["error"])
            return [("tavily", r["url"], r.get("content", "")) for r in result.get("results", [])]
        except Exception as e:
            print(f"Tavily deep MCP failed: {e}")
            return []

    async def _you_deep():
        try:
            result = await call_you_async(prompt, count=6)
            if result.get("error"):
                raise RuntimeError(result["error"])
            return [("you", r["url"], r.get("content", "")) for r in result.get("results", [])]
        except Exception as e:
            print(f"You.com deep MCP failed: {e}")
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

    for provider_results in results:
        if isinstance(provider_results, BaseException) or not provider_results:
            continue
        for _provider, url, text in provider_results:
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
