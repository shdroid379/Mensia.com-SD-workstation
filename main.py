import os
import json
import uuid
import asyncio
from datetime import date
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
# 3. API ENDPOINTS & LIVE BACKGROUND TASKS
# =====================================================================
intense_dive_tasks = {}

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
            "total_lifetime": 15,            # Visually satisfies the 15-chat condition
            "intense_dive_unlocked": True,    # Never disable or lock out Intense Dive
            "intense_dive_available": True
        },
        "exhausted": {
            "casual": False,
            "search": False,
            "deep research": False,
            "intense dive": False
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

# =====================================================================
# INTENSE DIVE WITH SOURCES
# =====================================================================
@app.post("/intense-dive")
async def intense_dive_endpoint(q: Question, user: dict = Depends(get_current_user)):
    task_id = str(uuid.uuid4())
    intense_dive_tasks[task_id] = {
        "status": "processing",
        "message": "DIVING IN...",
        "result": None,
        "sources": [],   # <-- now storing sources
        "error": None
    }

    def update_task_message(msg: str):
        if task_id in intense_dive_tasks:
            intense_dive_tasks[task_id]["message"] = msg

    async def run_pipeline():
        try:
            rephrased = await asyncio.to_thread(rephrase_if_followup, q.question, q.session_id)
            
            if q.include_academic:
                context, sources = await intense_dive.fetch_combined_dossier_with_academic_papers(rephrased, update_task_message)
            else:
                context, sources = await intense_dive.fetch_combined_dossier(rephrased, update_task_message)
                
            draft = await intense_dive.synthesize_with_mistral(rephrased, context, update_task_message)
            final_report = await intense_dive.audit_with_deepseek(rephrased, draft, update_task_message)

            # Store both result and sources
            intense_dive_tasks[task_id]["result"] = final_report
            intense_dive_tasks[task_id]["sources"] = sources
            intense_dive_tasks[task_id]["status"] = "completed"
            intense_dive_tasks[task_id]["message"] = "THE DOSSIER IS READY."

            histories.setdefault(q.session_id, []).append([q.question, final_report])
        except Exception as e:
            intense_dive_tasks[task_id]["status"] = "failed"
            intense_dive_tasks[task_id]["error"] = str(e)

    asyncio.create_task(run_pipeline())
    return {"task_id": task_id, "status": "processing"}

@app.get("/intense-dive/status/{task_id}")
async def get_intense_dive_status(task_id: str):
    task = intense_dive_tasks.get(task_id)
    if not task:
        return {"status": "not_found", "message": "Task not found"}
    return {
        "status": task.get("status"),
        "message": task.get("message"),
        "result": task.get("result"),
        "sources": task.get("sources", []),
        "error": task.get("error")
    }

# =====================================================================
# TEST HYPERBOLIC ENDPOINT (instant key verification)
# =====================================================================
@app.get("/test-hyperbolic")
async def test_hyperbolic():
    """Quick test to verify Hyperbolic API key and model availability."""
    from openai import OpenAI
    api_key = os.getenv("HYPERBOLIC_API_KEY")
    if not api_key:
        return {"error": "HYPERBOLIC_API_KEY not set in environment"}

    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.hyperbolic.xyz/v1",
            timeout=10.0
        )
        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3",
            messages=[{"role": "user", "content": "Say 'Hello, Mensia!'"}],
            temperature=0.0,
            max_tokens=10
        )
        return {
            "success": True,
            "response": response.choices[0].message.content,
            "usage": response.usage
        }
    except Exception as e:
        # Safely extract details
        status = getattr(e, 'status_code', None)
        body = getattr(e, 'body', None) or getattr(e, 'response', None)
        error_details = {
            "status_code": status,
            "body": str(body) if body is not None else None
        }
        return {"error": str(e), "details": error_details}
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