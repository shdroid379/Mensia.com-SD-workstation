import os
import json
import asyncio
from datetime import date
from fastapi import FastAPI, Request, Header, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from groq import Groq

import firebase_admin
from firebase_admin import credentials, auth

import pipeline
import ai_answer
import deep_multi_fetch

# =====================================================================
# 1. FIREBASE ADMIN INITIALIZATION
# =====================================================================
if not firebase_admin._apps:
    service_account_path = "firebase_service_account.json"
    if os.path.exists(service_account_path):
        cred = credentials.Certificate(service_account_path)
        firebase_admin.initialize_app(cred)
    else:
        # Fallback to environment variable if set securely in Render
        sa_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
        if sa_json:
            cred = credentials.Certificate(json.loads(sa_json))
            firebase_admin.initialize_app(cred)
        else:
            print("WARNING: No Firebase credentials found. Running in unauthenticated guest-only fallback mode.")

app = FastAPI()

# =====================================================================
# 2. TIER LIMITS & USAGE TRACKING
# =====================================================================
GUEST_LIMITS = {
    "casual": 25,
    "search": 5,
    "deep research": 1
}

AUTH_LIMITS = {
    "casual": 350,
    "search": 25,
    "deep research": 3
}

USAGE_FILE = "usage_logs.json"

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
    # Extracts the true IP address from Render's proxy
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "127.0.0.1"

async def get_current_user(request: Request, authorization: str = Header(None)):
    """Verifies Firebase token or falls back to IP-based guest tracking."""
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split("Bearer ")[1].strip()
        try:
            decoded_token = auth.verify_id_token(token)
            return {
                "id": f"usr_{decoded_token['uid']}",
                "email": decoded_token.get("email", ""),
                "is_authenticated": True
            }
        except Exception as e:
            print(f"Auth token verification failed: {e}")
    
    # Guest user tracked by IP
    client_ip = get_client_ip(request)
    return {
        "id": f"ip_{client_ip}",
        "email": None,
        "is_authenticated": False
    }

def check_and_increment_limit(user: dict, mode: str):
    limits = AUTH_LIMITS if user["is_authenticated"] else GUEST_LIMITS
    today = str(date.today())
    data = load_usage()
    
    # Auto-prune old days to save disk space
    for day in list(data.keys()):
        if day != today:
            del data[day]
            
    if today not in data:
        data[today] = {}
        
    user_id = user["id"]
    if user_id not in data[today]:
        data[today][user_id] = {"casual": 0, "search": 0, "deep research": 0}
        
    current_count = data[today][user_id].get(mode, 0)
    max_limit = limits.get(mode, 0)
    
    if current_count >= max_limit:
        save_usage(data)
        return False, current_count, max_limit
        
    data[today][user_id][mode] = current_count + 1
    save_usage(data)
    return True, current_count + 1, max_limit

# =====================================================================
# 3. API ENDPOINTS
# =====================================================================
@app.get("/limits")
async def get_limits(user: dict = Depends(get_current_user)):
    """Frontend calls this to sync disabled dropdown options."""
    limits = AUTH_LIMITS if user["is_authenticated"] else GUEST_LIMITS
    today = str(date.today())
    data = load_usage()
    user_usage = data.get(today, {}).get(user["id"], {})
    
    return {
        "is_authenticated": user["is_authenticated"],
        "email": user["email"],
        "limits": limits,
        "usage": {
            "casual": user_usage.get("casual", 0),
            "search": user_usage.get("search", 0),
            "deep research": user_usage.get("deep research", 0)
        },
        "exhausted": {
            "casual": user_usage.get("casual", 0) >= limits["casual"],
            "search": user_usage.get("search", 0) >= limits["search"],
            "deep research": user_usage.get("deep research", 0) >= limits["deep research"]
        }
    }

@app.get("/health")
def health_check():
    """Lightweight ping for Render/Cron to keep server alive."""
    return {"status": "alive"}

# Memory dictionary for active sessions
histories: dict[str, list[list[str]]] = {}

class Question(BaseModel):
    question: str
    mode: str = "casual"
    session_id: str

def rephrase_if_followup(question: str, session_id: str) -> str:
    """Uses a fast model to rewrite 'what is it' into 'what is X' based on history."""
    history = histories.get(session_id, [])
    if not history:
        return question

    try:
        client6 = Groq(api_key=os.getenv("GROQ_KEY"))
        response = client6.chat.completions.create(
            model="openai/gpt-oss-20b", # Ensure this matches your actual fast Groq model
            messages=[
                {"role": "system", "content": "You are a follow-up rephraser. Output only the rephrased standalone search query."},
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
async def ask(q: Question, request: Request, user: dict = Depends(get_current_user)):
    mode = q.mode.lower()
    allowed, count, max_lim = check_and_increment_limit(user, mode)
    
    if not allowed:
        auth_hint = "Sign in with a free account to unlock 350 Casual, 25 Search, and 3 Deep Research queries daily!" if not user["is_authenticated"] else "Please check the Plans page to upgrade."
        return {"error": f"Daily limit reached for {mode} ({count}/{max_lim}). {auth_hint}"}

    # Step 1: Rephrase
    rephrased = await asyncio.to_thread(rephrase_if_followup, q.question, q.session_id)
    
    # KILLSWITCH
    if await request.is_disconnected():
        return {"error": "Aborted"}

    # Step 2: Search Pipeline
    if mode == "search":
        context, sources = await asyncio.to_thread(pipeline.basic_search, rephrased)
    else:
        context, sources = "", []
        mode = "casual"

    # KILLSWITCH
    if await request.is_disconnected():
        return {"error": "Aborted"}

    # Step 3: Synthesis & Audit
    current_history = histories.get(q.session_id, [])
    answer = await asyncio.to_thread(ai_answer.ai_summary, rephrased, context, mode, current_history) or ""
    
    histories.setdefault(q.session_id, []).append([q.question, answer])
    return {"answer": answer, "sources": sources}

@app.post("/deep-research")
async def deep_research(q: Question, request: Request, user: dict = Depends(get_current_user)):
    mode = "deep research"
    allowed, count, max_lim = check_and_increment_limit(user, mode)
    
    if not allowed:
        auth_hint = "Create a free account to get 3 Deep Research queries daily!" if not user["is_authenticated"] else "Please check the Plans page for expanded limits."
        return {"error": f"Daily limit reached for Deep Research ({count}/{max_lim}). {auth_hint}"}

    # Step 1: Rephrase
    rephrased = await asyncio.to_thread(rephrase_if_followup, q.question, q.session_id)
    
    # KILLSWITCH
    if await request.is_disconnected():
        return {"error": "Aborted"}

    # Step 2: Parallel Web Search
    context, sources = await deep_multi_fetch.combined_research(rephrased)
    
    # KILLSWITCH
    if await request.is_disconnected():
        return {"error": "Aborted"}

    # Step 3: Synthesis & Audit
    current_history = histories.get(q.session_id, [])
    answer = await asyncio.to_thread(ai_answer.ai_summary, rephrased, context, mode, current_history) or ""

    histories.setdefault(q.session_id, []).append([q.question, answer])
    return {"answer": answer, "sources": sources}

# =====================================================================
# 4. STATIC FILES & SEO ROUTING
# =====================================================================
@app.get("/sitemap.xml")
def get_sitemap():
    return FileResponse("static/sitemap.xml", media_type="application/xml")

@app.get("/robots.txt")
def get_robots():
    return FileResponse("static/robots.txt", media_type="text/plain")

app.mount("/", StaticFiles(directory="static", html=True), name="static")