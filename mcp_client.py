import sys
import json
import asyncio
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

MCP_SERVERS_DIR = Path(__file__).resolve().parent / "mcp_servers"

SERVER_CONFIGS = {
    "exa": {
        "command": sys.executable,
        "args": [str(MCP_SERVERS_DIR / "exa_server.py")],
        "tool_name": "exa_search",
    },
    "tavily": {
        "command": sys.executable,
        "args": [str(MCP_SERVERS_DIR / "tavily_server.py")],
        "tool_name": "tavily_search",
    },
    "you": {
        "command": sys.executable,
        "args": [str(MCP_SERVERS_DIR / "you_server.py")],
        "tool_name": "you_search",
    },
}


async def _call_mcp_server(server_name: str, arguments: dict) -> str:
    """Connect to an MCP server via stdio, call a tool, return the result."""
    config = SERVER_CONFIGS[server_name]
    server_params = StdioServerParameters(
        command=config["command"],
        args=config["args"],
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(config["tool_name"], arguments=arguments)

            if result.content and len(result.content) > 0:
                text = result.content[0].text
                if text and text.strip():
                    return text
            return json.dumps({"error": "No content returned from MCP server"})


def call_exa_sync(query: str, count: int = 5, mode: str = "neural") -> dict:
    """Synchronous wrapper to call Exa MCP server."""
    result = asyncio.run(
        _call_mcp_server("exa", {"query": query, "count": count, "mode": mode})
    )
    try:
        return json.loads(result)
    except (json.JSONDecodeError, TypeError):
        return {"error": f"Invalid response from Exa MCP: {result[:200] if result else 'empty'}"}


def call_tavily_sync(
    query: str,
    count: int = 5,
    search_depth: str = "basic",
    include_answer: bool = False,
) -> dict:
    """Synchronous wrapper to call Tavily MCP server."""
    args = {"query": query, "count": count, "search_depth": search_depth}
    if include_answer:
        args["include_answer"] = True
    result = asyncio.run(_call_mcp_server("tavily", args))
    try:
        return json.loads(result)
    except (json.JSONDecodeError, TypeError):
        return {"error": f"Invalid response from Tavily MCP: {result[:200] if result else 'empty'}"}


def call_you_sync(query: str, count: int = 5) -> dict:
    """Synchronous wrapper to call You.com MCP server."""
    result = asyncio.run(
        _call_mcp_server("you", {"query": query, "count": count})
    )
    try:
        return json.loads(result)
    except (json.JSONDecodeError, TypeError):
        return {"error": f"Invalid response from You.com MCP: {result[:200] if result else 'empty'}"}


async def call_exa_async(query: str, count: int = 5, mode: str = "neural") -> dict:
    """Async call to Exa MCP server."""
    result = await _call_mcp_server("exa", {"query": query, "count": count, "mode": mode})
    try:
        return json.loads(result)
    except (json.JSONDecodeError, TypeError):
        return {"error": f"Invalid response from Exa MCP: {result[:200] if result else 'empty'}"}


async def call_tavily_async(
    query: str,
    count: int = 5,
    search_depth: str = "basic",
    include_answer: bool = False,
) -> dict:
    """Async call to Tavily MCP server."""
    args = {"query": query, "count": count, "search_depth": search_depth}
    if include_answer:
        args["include_answer"] = True
    result = await _call_mcp_server("tavily", args)
    try:
        return json.loads(result)
    except (json.JSONDecodeError, TypeError):
        return {"error": f"Invalid response from Tavily MCP: {result[:200] if result else 'empty'}"}


async def call_you_async(query: str, count: int = 5) -> dict:
    """Async call to You.com MCP server."""
    result = await _call_mcp_server("you", {"query": query, "count": count})
    try:
        return json.loads(result)
    except (json.JSONDecodeError, TypeError):
        return {"error": f"Invalid response from You.com MCP: {result[:200] if result else 'empty'}"}


def get_available_tools() -> list[dict]:
    """Return metadata about available MCP tools for LLM system prompts."""
    return [
        {
            "server": "exa",
            "tool": "exa_search",
            "description": "Neural semantic web search via Exa. Best for conceptual and AI-related queries.",
            "parameters": "query: str, count: int (1-10), mode: str (neural|keyword|deep)",
        },
        {
            "server": "tavily",
            "tool": "tavily_search",
            "description": "AI-optimized web search via Tavily. Best for research and factual queries.",
            "parameters": "query: str, count: int (1-10), search_depth: str (basic|advanced)",
        },
        {
            "server": "you",
            "tool": "you_search",
            "description": "Real-time web search via You.com. Best for news and current events.",
            "parameters": "query: str, count: int (1-20)",
        },
    ]
