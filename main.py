import os
import asyncio
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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
    history = histories.get(session_id, [])
    if not history:
        return question

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
    mode = q.mode.lower()
    rephrased = rephrase_if_followup(q.question, q.session_id)
    current_history = histories.get(q.session_id, [])

    if mode == "search":
        context, sources = pipeline.basic_search(rephrased)
    else:
        context = ""
        sources = []
        mode = "casual"

    answer = ai_answer.ai_summary(rephrased, context, mode, current_history) or ""
    histories.setdefault(q.session_id, []).append([q.question, answer])
    return {"answer": answer, "sources": sources}


@app.post("/deep-research")
async def deep_research(q: Question):
    rephrased = await asyncio.to_thread(rephrase_if_followup, q.question, q.session_id)
    current_history = histories.get(q.session_id, [])

    context, sources = await deep_multi_fetch.combined_research(rephrased)

    answer = await asyncio.to_thread(
        ai_answer.ai_summary, rephrased, context, "deep research", current_history
    ) or ""

    histories.setdefault(q.session_id, []).append([q.question, answer])
    return {"answer": answer, "sources": sources}


@app.get("/sitemap.xml")
def get_sitemap():
    return FileResponse("static/sitemap.xml", media_type="application/xml")


@app.get("/robots.txt")
def get_robots():
    return FileResponse("static/robots.txt", media_type="text/plain")


# Must be last — catches all other routes and serves the frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")