import os
import json
import asyncio
from datetime import date
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from groq import Groq

import pipeline
import ai_answer
import deep_multi_fetch

app = FastAPI()

# --- PERSISTENT LIMIT TRACKING ---
USAGE_FILE = "usage_logs.json"

LIMITS = {
    "casual": 350,
    "search": 25,
    "deep research": 2
}

def load_usage():
    if os.path.exists(USAGE_FILE):
        try:
            with open(USAGE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_usage(data):
    with open(USAGE_FILE, "w") as f:
        json.dump(data, f)

def get_client_ip(request: Request):
    # Render hides the real IP behind a proxy. We must read the 'X-Forwarded-For' header.
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host

def check_and_increment_limit(request: Request, mode: str) -> bool:
    ip = get_client_ip(request)
    today = str(date.today())
    data = load_usage()
    
    # Auto-delete yesterday's logs to save disk space
    keys = list(data.keys())
    for k in keys:
        if k != today:
            del data[k]
            
    if today not in data:
        data[today] = {}
    if ip not in data[today]:
        data[today][ip] = {"casual": 0, "search": 0, "deep research": 0}
        
    if data[today][ip].get(mode, 0) >= LIMITS.get(mode, 0):
        save_usage(data)
        return False
        
    data[today][ip][mode] += 1
    save_usage(data)
    return True

@app.get("/limits")
def get_limits(request: Request):
    """Frontend calls this to see which modes to grey out."""
    ip = get_client_ip(request)
    today = str(date.today())
    data = load_usage()
    user_usage = data.get(today, {}).get(ip, {})
    
    return {
        "casual": user_usage.get("casual", 0) >= LIMITS["casual"],
        "search": user_usage.get("search", 0) >= LIMITS["search"],
        "deep research": user_usage.get("deep research", 0) >= LIMITS["deep research"]
    }

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
                {"role": "system", "content": "You are a follow-up rephraser. Output only the rephrased prompt."},
                {"role": "user", "content": f"History: {history[-4:]}\nFollow-up: {question}\nRephrase it."}
            ],
            temperature=0.1,
            max_tokens=75
        )
        raw = response.choices[0].message.content
        return raw.strip() if raw else question
    except Exception:
        return question

@app.post("/ask")
async def ask(q: Question, request: Request):
    mode = q.mode.lower()
    
    if not check_and_increment_limit(request, mode):
        return {"error": f"Your IP has exhausted the daily limit for {mode} mode."}

    # Step 1: Rephrase
    rephrased = await asyncio.to_thread(rephrase_if_followup, q.question, q.session_id)
    
    # KILLSWITCH: If user aborted, stop here. Do not fire Exa/Tavily.
    if await request.is_disconnected():
        return {"error": "Aborted"}

    # Step 2: Search
    if mode == "search":
        context, sources = await asyncio.to_thread(pipeline.basic_search, rephrased)
    else:
        context, sources = "", []
        mode = "casual"

    # KILLSWITCH: If user aborted, stop here. Do not fire Gemini/Hyperbolic.
    if await request.is_disconnected():
        return {"error": "Aborted"}

    # Step 3: AI Synthesis & Audit
    current_history = histories.get(q.session_id, [])
    answer = await asyncio.to_thread(ai_answer.ai_summary, rephrased, context, mode, current_history) or ""
    
    histories.setdefault(q.session_id, []).append([q.question, answer])
    return {"answer": answer, "sources": sources}

@app.post("/deep-research")
async def deep_research(q: Question, request: Request):
    mode = "deep research"
    
    if not check_and_increment_limit(request, mode):
        return {"error": f"Your IP has exhausted the daily limit for {mode} mode."}

    # Step 1: Rephrase
    rephrased = await asyncio.to_thread(rephrase_if_followup, q.question, q.session_id)
    
    # KILLSWITCH: If user aborted, stop here. Do not fire 3x search APIs.
    if await request.is_disconnected():
        return {"error": "Aborted"}

    # Step 2: Concurrent Deep Search
    context, sources = await deep_multi_fetch.combined_research(rephrased)

    # KILLSWITCH: If user aborted during the 10-second search, do not fire LLMs.
    if await request.is_disconnected():
        return {"error": "Aborted"}

    # Step 3: AI Synthesis & Audit
    current_history = histories.get(q.session_id, [])
    answer = await asyncio.to_thread(ai_answer.ai_summary, rephrased, context, mode, current_history) or ""

    histories.setdefault(q.session_id, []).append([q.question, answer])
    return {"answer": answer, "sources": sources}

@app.get("/sitemap.xml")
def get_sitemap():
    return FileResponse("static/sitemap.xml", media_type="application/xml")

@app.get("/robots.txt")
def get_robots():
    return FileResponse("static/robots.txt", media_type="text/plain")

app.mount("/", StaticFiles(directory="static", html=True), name="static")