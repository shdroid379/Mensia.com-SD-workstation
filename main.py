import os
import json
import asyncio
from datetime import date, datetime
from fastapi import FastAPI, Request, Header, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from groq import Groq

import firebase_admin
from firebase_admin import credentials, auth

import pipeline
import ai_answer
import deep_multi_fetch
import intense_dive
import doc_builder

# =====================================================================
# 1. FIREBASE ADMIN INITIALIZATION
# =====================================================================
if not firebase_admin._apps:
    service_account_path = "firebase_service_account.json"
    if os.path.exists(service_account_path):
        cred = credentials.Certificate(service_account_path)
        firebase_admin.initialize_app(cred)
    else:
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
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "127.0.0.1"

async def get_current_user(request: Request, authorization: str = Header(None)):
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
    
    # Auto-prune old days while preserving persistent trackers
    for day in list(data.keys()):
        if day not in [today, "totals", "intense_dive_history"]:
            del data[day]
            
    if today not in data: data[today] = {}
    if "totals" not in data: data["totals"] = {}
    if "intense_dive_history" not in data: data["intense_dive_history"] = {}
        
    user_id = user["id"]
    if user_id not in data[today]:
        data[today][user_id] = {"casual": 0, "search": 0, "deep research": 0}
        
    user_total = data["totals"].get(user_id, 0)
    
    # Intense Dive Logic
    if mode == "intense dive":
        if not user["is_authenticated"]:
            return False, 0, 0, "Intense Dive requires a registered account."
        if user_total < 15:
            return False, 0, 0, f"Intense Dive unlocks after 15 total queries. You currently have {user_total}."
        
        last_idate = data["intense_dive_history"].get(user_id)
        if last_idate:
            last_d = datetime.strptime(last_idate, "%Y-%m-%d").date()
            if (date.today() - last_d).days < 7:
                return False, 0, 0, "Intense Dive is limited to 1 per week. Your cooldown is still active."
                
        # Passed checks: Record usage
        data["totals"][user_id] = user_total + 1
        data["intense_dive_history"][user_id] = today
        save_usage(data)
        return True, 1, 1, ""

    # Normal Modes Logic
    current_count = data[today][user_id].get(mode, 0)
    max_limit = limits.get(mode, 0)
    
    if current_count >= max_limit:
        auth_hint = "Sign in to unlock more." if not user["is_authenticated"] else "Upgrade plan to expand limits."
        return False, current_count, max_limit, f"Daily limit reached for {mode}. {auth_hint}"
        
    data[today][user_id][mode] = current_count + 1
    data["totals"][user_id] = user_total + 1
    save_usage(data)
    
    return True, current_count + 1, max_limit, ""

# =====================================================================
# 3. API ENDPOINTS
# =====================================================================
@app.get("/limits")
async def get_limits(user: dict = Depends(get_current_user)):
    limits = AUTH_LIMITS if user["is_authenticated"] else GUEST_LIMITS
    today = str(date.today())
    data = load_usage()
    user_usage = data.get(today, {}).get(user["id"], {})
    
    total_lifetime = data.get("totals", {}).get(user["id"], 0)
    last_idate = data.get("intense_dive_history", {}).get(user["id"])
    
    intense_dive_unlocked = total_lifetime >= 15 and user["is_authenticated"]
    intense_dive_available = False
    
    if intense_dive_unlocked:
        if not last_idate:
            intense_dive_available = True
        else:
            last_d = datetime.strptime(last_idate, "%Y-%m-%d").date()
            if (date.today() - last_d).days >= 7:
                intense_dive_available = True

    return {
        "is_authenticated": user["is_authenticated"],
        "email": user["email"],
        "limits": limits,
        "usage": {
            "casual": user_usage.get("casual", 0),
            "search": user_usage.get("search", 0),
            "deep research": user_usage.get("deep research", 0),
            "total_lifetime": total_lifetime,
            "intense_dive_unlocked": intense_dive_unlocked,
            "intense_dive_available": intense_dive_available
        },
        "exhausted": {
            "casual": user_usage.get("casual", 0) >= limits.get("casual", 0),
            "search": user_usage.get("search", 0) >= limits.get("search", 0),
            "deep research": user_usage.get("deep research", 0) >= limits.get("deep research", 0),
            "intense dive": not intense_dive_available
        }
    }

@app.get("/health")
def health_check():
    return {"status": "alive"}

histories: dict[str, list[list[str]]] = {}

class Question(BaseModel):
    question: str
    mode: str = "casual"
    session_id: str
    include_academic: bool = False

def rephrase_if_followup(question: str, session_id: str) -> str:
    history = histories.get(session_id, [])
    if not history: return question
    try:
        client6 = Groq(api_key=os.getenv("GROQ_KEY"))
        response = client6.chat.completions.create(
            model="llama3-8b-8192", 
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
    allowed, count, max_lim, err = check_and_increment_limit(user, mode)
    if not allowed: return {"error": err}

    rephrased = await asyncio.to_thread(rephrase_if_followup, q.question, q.session_id)
    if await request.is_disconnected(): return {"error": "Aborted"}

    if mode == "search":
        context, sources = await asyncio.to_thread(pipeline.basic_search, rephrased)
    else:
        context, sources = "", []
        mode = "casual"

    if await request.is_disconnected(): return {"error": "Aborted"}

    current_history = histories.get(q.session_id, [])
    answer = await asyncio.to_thread(ai_answer.ai_summary, rephrased, context, mode, current_history) or ""
    
    histories.setdefault(q.session_id, []).append([q.question, answer])
    return {"answer": answer, "sources": sources}

@app.post("/deep-research")
async def deep_research(q: Question, request: Request, user: dict = Depends(get_current_user)):
    mode = "deep research"
    allowed, count, max_lim, err = check_and_increment_limit(user, mode)
    if not allowed: return {"error": err}

    rephrased = await asyncio.to_thread(rephrase_if_followup, q.question, q.session_id)
    if await request.is_disconnected(): return {"error": "Aborted"}

    context, sources = await deep_multi_fetch.combined_research(rephrased)
    if await request.is_disconnected(): return {"error": "Aborted"}

    current_history = histories.get(q.session_id, [])
    answer = await asyncio.to_thread(ai_answer.ai_summary, rephrased, context, mode, current_history) or ""

    histories.setdefault(q.session_id, []).append([q.question, answer])
    return {"answer": answer, "sources": sources}

@app.post("/intense-dive")
async def intense_dive_endpoint(q: Question, request: Request, user: dict = Depends(get_current_user)):
    mode = "intense dive"
    allowed, count, max_lim, err = check_and_increment_limit(user, mode)
    if not allowed: return {"error": err}

    rephrased = await asyncio.to_thread(rephrase_if_followup, q.question, q.session_id)
    if await request.is_disconnected(): return {"error": "Aborted"}

    if q.include_academic:
        context = await intense_dive.fetch_combined_dossier_with_academic_papers(rephrased)
    else:
        context = await intense_dive.fetch_combined_dossier(rephrased)
        
    if await request.is_disconnected(): return {"error": "Aborted"}

    draft = await intense_dive.synthesize_with_mistral(rephrased, context)
    final_report = await intense_dive.audit_with_deepseek(rephrased, draft)

    histories.setdefault(q.session_id, []).append([q.question, final_report])
    return {"answer": final_report, "sources": []}

# =====================================================================
# 4. DOCUMENT EXPORT ENDPOINTS
# =====================================================================
class ExportRequest(BaseModel):
    markdown: str

@app.post("/export/pdf")
def export_pdf(req: ExportRequest):
    buf = doc_builder.build_pdf_buffer(req.markdown)
    return StreamingResponse(buf, media_type="application/pdf", headers={"Content-Disposition": "attachment; filename=Mensia_Dossier.pdf"})

@app.post("/export/docx")
def export_docx(req: ExportRequest):
    buf = doc_builder.build_docx_buffer(req.markdown)
    return StreamingResponse(buf, media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document", headers={"Content-Disposition": "attachment; filename=Mensia_Dossier.docx"})

@app.post("/export/md")
def export_md(req: ExportRequest):
    buf = doc_builder.build_markdown_buffer(req.markdown)
    return StreamingResponse(buf, media_type="text/markdown", headers={"Content-Disposition": "attachment; filename=Mensia_Dossier.md"})

# =====================================================================
# 5. STATIC FILES & SEO ROUTING
# =====================================================================
@app.get("/sitemap.xml")
def get_sitemap():
    return FileResponse("static/sitemap.xml", media_type="application/xml")

@app.get("/robots.txt")
def get_robots():
    return FileResponse("static/robots.txt", media_type="text/plain")

app.mount("/", StaticFiles(directory="static", html=True), name="static")