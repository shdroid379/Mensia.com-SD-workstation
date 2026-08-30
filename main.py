import os
import asyncio
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from groq import Groq

import pipeline
import ai_answer
import deep_multi_fetch

app = FastAPI()


@app.get("/health")
def health_check():
    return {"status": "alive"}


histories: dict[str, list[list[str]]] = {}


class Question(BaseModel):
    question: str
    mode: str = "casual"
    session_id: str


def rephrase_if_followup(question: str, session_id: str) -> str:
    """
    Sync function — safe to call directly from sync routes,
    or via asyncio.to_thread from async routes.
    """
    history = histories.get(session_id, [])
    if not history:
        return question  # first message in this session, nothing to rephrase against

    try:
        client6 = Groq(api_key=os.getenv("GROQ_KEY"))
        response = client6.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a follow-up rephraser. Read the history, interpret context, "
                        "and rephrase the follow-up into a standalone search-engine-ready query. "
                        "Output only the rephrased prompt — no preamble, no labels, no explanation."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"History: {history[-4:]}\n"
                        f"Follow-up: {question}\n"
                        "Rephrase the follow-up based on the context."
                    )
                }
            ],
            temperature=0.1,
            max_tokens=75,
            include_reasoning=False,
            reasoning_effort="low"
        )
        raw = response.choices[0].message.content
        return raw.strip() if raw else question
    except Exception as e:
        print(f"Groq Rephraser failed: {e}. Using original query.")
        return question


@app.post("/ask")
def ask(q: Question):
    """
    Sync route — FastAPI automatically runs this in a thread pool.
    All blocking calls (Groq, Exa/Tavily/Linkup, Gemini, Hyperbolic)
    are safe here without any extra wrapping.
    """
    mode = q.mode.lower()
    rephrased = rephrase_if_followup(q.question, q.session_id)
    current_history = histories.get(q.session_id, [])

    if mode == "search":
        context = pipeline.basic_search(rephrased)
    else:
        context = ""
        mode = "casual"

    answer = ai_answer.ai_summary(rephrased, context, mode, current_history) or ""
    histories.setdefault(q.session_id, []).append([q.question, answer])
    return {"answer": answer}


@app.post("/deep-research")
async def deep_research(q: Question):
    """
    Async route — needed because combined_research() is a genuine coroutine.
    Every blocking sync call must be wrapped in asyncio.to_thread()
    to avoid freezing the event loop.
    """
    # rephrase_if_followup is a blocking Groq call — thread it
    rephrased = await asyncio.to_thread(rephrase_if_followup, q.question, q.session_id)
    current_history = histories.get(q.session_id, [])

    # combined_research is genuinely async — await directly
    context = await deep_multi_fetch.combined_research(rephrased)

    # ai_summary is blocking (Gemini + Hyperbolic) — thread it
    answer = await asyncio.to_thread(
        ai_answer.ai_summary, rephrased, context, "deep research", current_history
    ) or ""

    histories.setdefault(q.session_id, []).append([q.question, answer])
    return {"answer": answer}

from fastapi.responses import FileResponse

@app.get("/sitemap.xml")
def get_sitemap():
    return FileResponse("static/sitemap.xml", media_type="application/xml")

@app.get("/robots.txt")
def get_robots():
    return FileResponse("static/robots.txt", media_type="text/plain")
# Must be last — catches all other routes and serves the frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")