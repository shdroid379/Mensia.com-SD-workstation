import os
import json
import asyncio
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("tavily-search")


@mcp.tool()
async def tavily_search(
    query: str,
    count: int = 5,
    search_depth: str = "basic",
    include_answer: bool = False,
) -> str:
    """Search the web using Tavily search engine optimized for AI agents.

    Args:
        query: The search query to execute.
        count: Number of results to return (1-10). Default 5.
        search_depth: 'basic' for fast results, 'advanced' for deeper results. Default 'basic'.
        include_answer: Whether to include a pre-synthesized answer. Default False.

    Returns:
        JSON string with search results containing url and content for each result.
    """
    from tavily import AsyncTavilyClient

    api_key = os.getenv("TAVILY_KEY")
    if not api_key:
        return json.dumps({"error": "TAVILY_KEY not configured"})

    client = AsyncTavilyClient(api_key=api_key)

    try:
        search_kwargs = {
            "query": query,
            "max_results": min(count, 10),
            "search_depth": search_depth,
            "chunks_per_source": "auto",
        }

        if include_answer:
            search_kwargs["include_answer"] = "advanced"

        response = await client.search(**search_kwargs)

        results = []
        for item in response.get("results", []):
            results.append({
                "url": item.get("url", ""),
                "title": item.get("title", ""),
                "content": item.get("content", ""),
            })

        answer = response.get("answer", None)
        return json.dumps({
            "results": results,
            "count": len(results),
            "answer": answer,
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


if __name__ == "__main__":
    mcp.run(transport="stdio")
