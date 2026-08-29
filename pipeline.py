import os
from pathlib import Path
from dotenv import load_dotenv
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)
from exa_py import Exa
from tavily import TavilyClient
from linkup import LinkupClient



class Search_pipeline():
    def __init__(self, prompt):
        self.prompt = prompt
    def exa_search(self):
        client1 = Exa(api_key=os.getenv("EXA_API_KEY"))
        response = client1.search(self.prompt, num_results=5, contents={"text": {"max_characters": 4500}}, type= "neural")
        return [(item.url, item.text or "") for item in response.results]
    def tavily_search(self):
        client2 = TavilyClient(api_key=os.getenv("TAVILY_KEY"))
        answer = client2.search(
        query=self.prompt,
        max_results=5,
        search_depth="basic",
        chunks_per_source="auto"          
        )
        return [(item['url'], item['content']) for item in answer["results"]]
    def tavily_deep_research(self):
            client2 = TavilyClient(api_key=os.getenv("TAVILY_KEY"))
            answer = client2.search(
            query=self.prompt,
            max_results=7,
            search_depth="advanced",
            chunks_per_source="auto"          
            )
            return [(item['url'], item['content']) for item in answer["results"]]
    def linkup_search(self):
        client3 = LinkupClient(api_key=os.getenv("LINKUP_KEY"))
        output = client3.search(
            query=self.prompt,
            depth="standard",
            output_type="searchResults",
            max_results=5
        )            
        return [(item.url, item.content) for item in output.results]

def _format(items: list[tuple[str, str]]):
    combined = ""
    for url, text in items:
        combined += f"Source({url}):\n{text}\n\n"
    return combined

def basic_search(prompt: str):
    search = Search_pipeline(prompt)
    try:
        exa_result = search.exa_search()
        if exa_result:
            return _format(exa_result)
    except Exception as e:
        print(f"Exa failed due to {e}, trying Tavily now...")

    try:
        tavily_result = search.tavily_search()
        if tavily_result:
            return _format(tavily_result)
    except Exception as e:
        print(f"Tavily failed too due to {e}, trying Linkup now...")

    try:
        linkup_result = search.linkup_search()
        if linkup_result:
            return _format(linkup_result)
    except Exception as e:
        print(f"Linkup failed too due to {e}.")

    return "All search engines failed. Sorry for inconvenience, please try again later."



