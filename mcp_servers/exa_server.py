import os
import json
import asyncio
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("exa-search")


@mcp.tool()
async def exa_search(query: str, count: int = 5, mode: str = "neural") -> str:
    """Search the web using Exa neural search engine.

    Args:
        query: The search query to execute.
        count: Number of results to return (1-10). Default 5.
        mode: Search mode - 'neural' for semantic search, 'keyword' for keyword search, 'deep' for deep research. Default 'neural'.

    Returns:
        JSON string with search results containing url, title, and text for each result.
    """
    from exa_py import AsyncExa

    api_key = os.getenv("EXA_API_KEY")
    if not api_key:
        return json.dumps({"error": "EXA_API_KEY not configured"})

    client = AsyncExa(api_key=api_key)

    try:
        search_kwargs = {
            "query": query,
            "num_results": min(count, 10),
            "contents": {"text": {"max_characters": 4500}},
        }

        if mode == "deep":
            search_kwargs["type"] = "deep"
            search_kwargs["contents"] = {"text": {"max_characters": 5000}}
        elif mode == "keyword":
            search_kwargs["type"] = "keyword"
        else:
            search_kwargs["type"] = "neural"

        response = await client.search(**search_kwargs)

        results = []
        for item in response.results:
            results.append({
                "url": item.url,
                "title": getattr(item, "title", ""),
                "text": item.text or "",
            })

        return json.dumps({"results": results, "count": len(results)})
    except Exception as e:
        return json.dumps({"error": str(e)})


if __name__ == "__main__":
    mcp.run(transport="stdio")
