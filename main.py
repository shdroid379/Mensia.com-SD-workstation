import os
import json
import asyncio
from datetime import date
from fastapi import FastAPI, Request, Header, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from groq import Groq

import firebase_admin
from firebase_admin import credentials, auth
import logging
from datetime import datetime, timezone

# Configure logging so Render captures it instantly
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import pipeline
import ai_answer
import deep_multi_fetch
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
            print("Running in guest-ready Firebase fallback mode.")

app = FastAPI()

# =====================================================================
# 2. LIMITS & USAGE TRACKING (UNLOCKED FOR TESTING)
# =====================================================================
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
    # UNRESTRICTED FOR ACTIVE TESTING: Always returns allowed
    return True, 1, 9999, ""

# =====================================================================
# 2.5 CENTRALIZED LOGGING HELPER
# =====================================================================
def log_interaction(user: dict, session_id: str, mode: str, question: str, answer: str):
    timestamp = datetime.now(timezone.utc).isoformat()
    # Truncate to 150 chars to prevent terminal flood on Render
    short_q = question[:150] + "..." if len(question) > 150 else question
    short_a = answer[:150] + "..." if len(answer) > 150 else answer
    user_identifier = user.get("email") or user.get("id") or "Unknown"
    
    logger.info(
        f"TIMESTAMP: {timestamp} | "
        f"USER: {user_identifier} | "
        f"SESSION: {session_id} | "
        f"MODE: {mode} | "
        f"Q: '{short_q}' | "
        f"A: '{short_a}'"
    )

# =====================================================================
# 3. API ENDPOINTS
# =====================================================================

@app.get("/limits")
async def get_limits(user: dict = Depends(get_current_user)):
    return {
        "is_authenticated": user["is_authenticated"],
        "email": user["email"],
        "limits": {"casual": 350, "search": 25, "deep research": 3},
        "usage": {
            "casual": 0,
            "search": 0,
            "deep research": 0,
            "total_lifetime": 15
        },
        "exhausted": {
            "casual": False,
            "search": False,
            "deep research": False
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
        context, sources = await pipeline.basic_search(rephrased)
    else:
        context, sources = "", []
        mode = "casual"

    if await request.is_disconnected(): return {"error": "Aborted"}

    current_history = histories.get(q.session_id, [])
    answer = await asyncio.to_thread(ai_answer.ai_summary, rephrased, context, mode, current_history) or ""
    
    histories.setdefault(q.session_id, []).append([q.question, answer])
    
    # Log the interaction
    log_interaction(user, q.session_id, mode, q.question, answer)
    
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
    
    # Log the interaction
    log_interaction(user, q.session_id, mode, q.question, answer)
    
    return {"answer": answer, "sources": sources}

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

# Catch-all routes for the Single Page Application
@app.get("/workspace")
@app.get("/plans")
@app.get("/guide")
@app.get("/account")
def serve_spa_pages():
    return FileResponse("static/index.html")

app.mount("/", StaticFiles(directory="static", html=True), name="static")