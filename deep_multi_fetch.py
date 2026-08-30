import asyncio
import os
import urllib.parse
from pipeline import Search_pipeline


async def combined_research(prompt):
    search_up = Search_pipeline(prompt)
    
    results = await asyncio.gather(
        asyncio.to_thread(search_up.exa_search),
        asyncio.to_thread(search_up.tavily_deep_research),
        asyncio.to_thread(search_up.linkup_search),
        return_exceptions=True
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