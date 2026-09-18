import os
import json
import asyncio
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("you-search")


@mcp.tool()
async def you_search(
    query: str,
    count: int = 5,
    livecrawl: str = "web",
    livecrawl_formats: str = "markdown",
) -> str:
    """Search the web using You.com search API.

    Args:
        query: The search query to execute.
        count: Number of results to return (1-20). Default 5.
        livecrawl: Crawl mode - 'web', 'news', or 'all'. Default 'web'.
        livecrawl_formats: Output format - 'markdown' or 'html'. Default 'markdown'.

    Returns:
        JSON string with search results containing url, title, and snippets for each result.
    """
    import httpx

    api_key = os.getenv("YOU_API_KEY")
    if not api_key:
        return json.dumps({"error": "YOU_API_KEY not configured"})

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            payload = {
                "query": query,
                "count": min(count, 20),
                "livecrawl": livecrawl,
                "livecrawl_formats": [livecrawl_formats],
            }

            response = await client.post(
                "https://ydc-index.io/v1/search",
                headers={
                    "X-API-Key": api_key,
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            results = []
            web_results = data.get("results", {}).get("web", [])
            for item in web_results:
                content_parts = []
                if item.get("snippets"):
                    content_parts.extend(item["snippets"])
                elif item.get("description"):
                    content_parts.append(item["description"])

                results.append({
                    "url": item.get("url", ""),
                    "title": item.get("title", ""),
                    "content": "\n".join(content_parts) if content_parts else "",
                })

            return json.dumps({"results": results, "count": len(results)})
        except Exception as e:
            return json.dumps({"error": str(e)})


if __name__ == "__main__":
    mcp.run(transport="stdio")
