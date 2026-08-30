import asyncio
import os
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

    for item in results:
        if isinstance(item, BaseException) or not item:
            continue
        for url, text in item:
            if url in seen_urls:
                continue
            seen_urls.add(url)
            combined_content += f"Source: {url} \n {text}\n\n"
    return combined_content or "All search engines failed, try again later.."




