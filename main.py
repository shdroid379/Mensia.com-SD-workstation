import os
import json
import hmac
import asyncio
from datetime import date
from fastapi import FastAPI, Request, Header, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from groq import Groq

import firebase_admin
from firebase_admin import credentials, auth
import logging
from datetime import datetime, timezone
from fastapi import FastAPI, Request, Header, Depends, UploadFile, File
import httpx
from pathlib import Path
from dotenv import load_dotenv
# Configure logging so Render captures it instantly
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load .env before any module reads a key, so a plain `uvicorn main:app` run has
# the same configuration as the deployed service.
load_dotenv(Path(__file__).resolve().parent / ".env", override=False)

import pipeline
import ai_answer
import deep_multi_fetch
import doc_builder
import quotas
import vin_jobs

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
            logger.warning("Firebase credentials are not configured; authenticated requests will be unavailable.")

app = FastAPI()

# The local access-code generator opens as a file:// page and calls these APIs.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["null", "file://", "http://127.0.0.1:8000", "http://localhost:8000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BUCKETS = quotas.BUCKETS

# =====================================================================
# 2. AUTH + REAL, ENFORCED LIMITS
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
    raise HTTPException(status_code=401, detail={
        "error": "Please sign in or create an account to use Mensia."
    })

def _db_or_none():
    try:
        return quotas.db_client()
    except Exception as exc:
        logger.warning("Firestore unavailable: %s", exc)
        return None


def _shape(snap: dict | None, extra: dict) -> dict:
    """Uniform /limits body, whether or not Firestore answered."""
    if snap is None:
        snap = quotas.snapshot(None)
    rows = snap["rows"]
    exhausted = {b: rows[b]["exhausted"] for b in BUCKETS}
    return {
        **extra,
        "plan": snap["plan"],
        "window": snap["window"],
        "week": snap["week"],
        "rows": rows,
        "limits": {b: rows[b]["cap"] for b in BUCKETS},
        "usage": {b: rows[b]["used"] for b in BUCKETS},
        "exhausted": exhausted,
        "any_exhausted": any(exhausted.values()),
    }


def _unavailable_payload(user: dict) -> dict:
    return _shape(None, {
        "is_authenticated": user["is_authenticated"],
        "email": user["email"],
        "offline": True,
    })


def _limits_payload(user: dict) -> dict:
    db = _db_or_none()
    if db is None:
        return _unavailable_payload(user)
    try:
        doc = quotas.get_or_create(db, user["id"], user["email"])
    except Exception as exc:
        logger.warning("limits read failed for %s: %s", user["id"], exc)
        return _unavailable_payload(user)
    return _shape(quotas.snapshot(doc), {
        "is_authenticated": user["is_authenticated"],
        "email": user["email"],
        "offline": False,
    })


def _consume(user: dict, mode: str, variant: str | None = None):
    """Returns (allowed, message, payload). Fails open with no charge if the DB is down."""
    db = _db_or_none()
    if db is None:
        return True, "", _unavailable_payload(user)
    try:
        ok, msg, snap = quotas.check_and_consume(
            db, user["id"], user["email"], mode, variant)
    except Exception as exc:
        logger.warning("quota check failed for %s: %s", user["id"], exc)
        return True, "", _unavailable_payload(user)
    return ok, msg, _shape(snap, {
        "is_authenticated": user["is_authenticated"],
        "email": user["email"],
        "offline": False,
    })

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
    return _limits_payload(user)


MAX_AUDIO_BYTES = 15 * 1024 * 1024


@app.post("/api/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """Transcribe a short browser recording with Deepgram Nova-3."""
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        logger.error("DEEPGRAM_API_KEY is not set; voice input is unavailable.")
        raise HTTPException(
            status_code=503,
            detail={"error": "Voice input is not configured on this server."},
        )

    content = await audio.read()
    if not content:
        raise HTTPException(status_code=400, detail={"error": "No audio was received."})
    if len(content) > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=413,
            detail={"error": "That recording is too long. Keep it under a minute."},
        )

    # Nova-3 with language auto-detection: the app is used in several languages.
    url = (
        "https://api.deepgram.com/v1/listen"
        "?model=nova-3&detect_language=true&punctuate=true&smart_format=true"
    )
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": (audio.content_type or "audio/webm").split(";")[0],
    }

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
            response = await client.post(url, headers=headers, content=content)
        if response.status_code >= 400:
            logger.error(
                "Deepgram rejected transcription (%s): %s",
                response.status_code,
                response.text[:500],
            )
            raise HTTPException(
                status_code=502,
                detail={"error": "Transcription failed. Please try again."},
            )
        data = response.json()
        try:
            transcript = data["results"]["channels"][0]["alternatives"][0]["transcript"]
        except (KeyError, IndexError, TypeError):
            transcript = ""
        return {"transcript": (transcript or "").strip()}
    except HTTPException:
        raise
    except httpx.TimeoutException:
        logger.error("Deepgram transcription timed out after 120s")
        raise HTTPException(
            status_code=504,
            detail={"error": "Transcription took too long. Try a shorter recording."},
        )
    except Exception as e:
        logger.error(f"Deepgram Nova-3 transcription failed: {e}")
        raise HTTPException(
            status_code=502,
            detail={"error": "Transcription failed. Please try again."},
        )


@app.get("/health")
def health_check():
    return {"status": "alive"}

histories: dict[str, list[list[str]]] = {}

class Question(BaseModel):
    question: str
    mode: str = "search"
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
    if mode != "search":
        raise HTTPException(status_code=400, detail={"error": "Unsupported mode."})

    allowed, msg, payload = _consume(user, mode)
    if not allowed:
        raise HTTPException(status_code=429, detail={"error": msg, "limits": payload})

    rephrased = await asyncio.to_thread(rephrase_if_followup, q.question, q.session_id)
    if await request.is_disconnected(): return {"error": "Aborted"}

    context, sources = await pipeline.basic_search(rephrased)

    if await request.is_disconnected(): return {"error": "Aborted"}

    answer = await asyncio.to_thread(ai_answer.ai_summary, rephrased, context, mode) or ""

    histories.setdefault(q.session_id, []).append([q.question, answer])

    log_interaction(user, q.session_id, mode, q.question, answer)

    return {"answer": answer, "sources": sources, "limits": payload}

@app.post("/deep-research")
async def deep_research(q: Question, request: Request, user: dict = Depends(get_current_user)):
    mode = "deep research"
    allowed, msg, payload = _consume(user, mode)
    if not allowed:
        raise HTTPException(status_code=429, detail={"error": msg, "limits": payload})

    rephrased = await asyncio.to_thread(rephrase_if_followup, q.question, q.session_id)
    if await request.is_disconnected(): return {"error": "Aborted"}

    context, sources = await deep_multi_fetch.combined_research(rephrased)
    if await request.is_disconnected(): return {"error": "Aborted"}

    answer = await asyncio.to_thread(ai_answer.ai_summary, rephrased, context, mode) or ""

    histories.setdefault(q.session_id, []).append([q.question, answer])

    log_interaction(user, q.session_id, mode, q.question, answer)

    return {"answer": answer, "sources": sources, "limits": payload}

# =====================================================================
# 3.5 VIN - BACKGROUND RESEARCH ENGINE
# =====================================================================
class VinRequest(BaseModel):
    question: str
    variant: str = "base"
    session_id: str

@app.post("/vin")
async def start_vin(req: VinRequest, user: dict = Depends(get_current_user)):
    if not user["is_authenticated"]:
        raise HTTPException(status_code=401, detail={
            "error": "Vin runs against your account, so it needs a sign-in."})

    db = _db_or_none()
    if db is None:
        raise HTTPException(status_code=503, detail={
            "error": "Vin is briefly unavailable. Try again in a moment."})

    variant = "investigator" if req.variant.lower().startswith("inv") else "base"
    allowed, msg, payload = _consume(user, "vin", variant)
    if not allowed:
        raise HTTPException(status_code=429, detail={"error": msg, "limits": payload})

    job = vin_jobs.create_job(db, user, req.question, variant)
    vin_jobs.submit(db, job["job_id"], req.question, job["variant"])
    log_interaction(user, req.session_id, f"vin {variant}", req.question, f"job {job['job_id']}")
    return {"job": vin_jobs.public_view(job), "limits": payload}


@app.get("/vin/jobs")
async def my_vin_jobs(user: dict = Depends(get_current_user)):
    db = _db_or_none()
    if db is None:
        return {"jobs": []}
    try:
        return {"jobs": vin_jobs.list_mine(db, user["id"])}
    except Exception as exc:
        logger.warning("job list failed for %s: %s", user["id"], exc)
        return {"jobs": []}


@app.get("/vin/{job_id}")
async def poll_vin(job_id: str, user: dict = Depends(get_current_user)):
    db = _db_or_none()
    if db is None:
        raise HTTPException(status_code=503, detail={"error": "Vin is briefly unavailable."})
    job = vin_jobs.load_mine(db, job_id, user["id"])
    if job is None:
        raise HTTPException(status_code=404, detail={"error": "No such run."})
    return {"job": job}

# =====================================================================
# 3.6 ACCESS CODES
# =====================================================================
class RedeemRequest(BaseModel):
    code: str

@app.post("/access/redeem")
async def redeem_access(req: RedeemRequest, user: dict = Depends(get_current_user)):
    if not user["is_authenticated"]:
        raise HTTPException(status_code=401, detail={
            "error": "Sign in before redeeming an access code."})
    db = _db_or_none()
    if db is None:
        raise HTTPException(status_code=503, detail={
            "error": "Codes can't be checked right now. Try again shortly."})
    ok, msg, snap = quotas.redeem_code(db, user["id"], user["email"], req.code)
    if not ok:
        raise HTTPException(status_code=400, detail={"error": msg})
    return {"message": msg, "limits": _shape(snap, {
        "is_authenticated": user["is_authenticated"],
        "email": user["email"],
        "offline": False,
    })}


class AdminCodeRequest(BaseModel):
    secret: str
    kind: str = "extra"
    count: int = 1

def _admin_ok(secret: str) -> bool:
    expected = os.getenv("MENSIA_ADMIN_SECRET", "")
    return bool(expected) and hmac.compare_digest(secret or "", expected)

@app.post("/admin/access/codes")
async def admin_generate_codes(req: AdminCodeRequest):
    if not _admin_ok(req.secret):
        raise HTTPException(status_code=403, detail={"error": "Invalid admin secret."})
    db = _db_or_none()
    if db is None:
        raise HTTPException(status_code=503, detail={"error": "Firestore unavailable."})
    kind = "special" if req.kind.lower().startswith("spec") else "extra"
    codes = []
    for _ in range(max(1, min(req.count, 50))):
        code, hashed = quotas.generate_code(kind)
        quotas.store_code(db, kind, code, hashed)
        codes.append(code)
    return {"kind": kind, "codes": codes}


@app.post("/admin/access/list")
async def admin_list_codes(req: AdminCodeRequest):
    if not _admin_ok(req.secret):
        raise HTTPException(status_code=403, detail={"error": "Invalid admin secret."})
    db = _db_or_none()
    if db is None:
        raise HTTPException(status_code=503, detail={"error": "Firestore unavailable."})
    return {"codes": quotas.list_codes(db)}


@app.on_event("startup")
async def _resume_interrupted_vin_runs():
    db = _db_or_none()
    if db is None:
        return
    try:
        await vin_jobs.recover_orphans(db)
    except Exception as exc:
        logger.warning("Vin orphan recovery failed: %s", exc)

# =====================================================================
# 4. DOCUMENT EXPORT ENDPOINTS
# =====================================================================
class ExportRequest(BaseModel):
    markdown: str

@app.post("/export/pdf")
def export_pdf(req: ExportRequest, user: dict = Depends(get_current_user)):
    buf = doc_builder.build_pdf_buffer(req.markdown)
    return StreamingResponse(buf, media_type="application/pdf", headers={"Content-Disposition": "attachment; filename=Mensia_Dossier.pdf"})

@app.post("/export/docx")
def export_docx(req: ExportRequest, user: dict = Depends(get_current_user)):
    buf = doc_builder.build_docx_buffer(req.markdown)
    return StreamingResponse(buf, media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document", headers={"Content-Disposition": "attachment; filename=Mensia_Dossier.docx"})

@app.post("/export/md")
def export_md(req: ExportRequest, user: dict = Depends(get_current_user)):
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
@app.get("/increased-access")
@app.get("/about-vin")
@app.get("/extra-access")
def serve_spa_pages():
    return FileResponse("static/index.html")

app.mount("/", StaticFiles(directory="static", html=True), name="static")
