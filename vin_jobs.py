"""Vin runs as a background job so a 10-35 minute pipeline never blocks an HTTP request.

State lives in Firestore, never on disk: Render's filesystem is wiped on every
redeploy, restart, or spin-down. A run only needs the temp directory for as long as
the process is alive.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import quotas
import vin

logger = logging.getLogger(__name__)

JOBS = "mensia_vin_jobs"

# Progress is reported as a percentage plus a plain-language phase. The internal
# pipeline stages are never named to the client.
BASE_SEQUENCE = ["plan", "retrieve", "rerank", "ledger", "dossier", "verify", "finalize"]
INVESTIGATOR_SEQUENCE = ["plan", "retrieve", "rerank", "ledger", "dossier",
                         "redteam", "counter", "patch", "verify", "finalize"]

PHASE_LABELS = {
    "plan": "Framing the question",
    "retrieve": "Reading the live web",
    "rerank": "Weighing the sources",
    "ledger": "Cross-checking every figure against its source",
    "dossier": "Writing the report",
    "redteam": "Challenging the draft for weak claims",
    "counter": "Hunting down counter-evidence",
    "patch": "Revising what the challenge caught",
    "verify": "Auditing every claim, one by one, against its sources",
    "finalize": "Finalising the dossier",
}

ETA_MINUTES = {"base": (10, 20), "investigator": (15, 35)}

MAX_CONCURRENT = int(os.getenv("VIN_MAX_CONCURRENT", "1"))
MAX_ATTEMPTS = int(os.getenv("VIN_MAX_ATTEMPTS", "2"))
HEARTBEAT_STALE_S = 150

_semaphore: asyncio.Semaphore | None = None
_tasks: dict[str, asyncio.Task] = {}


def _sem() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    return _semaphore


def _col(db):
    return db.collection(JOBS)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _variant(raw: str | None) -> str:
    return "investigator" if (raw or "").lower().startswith("inv") else "base"


def _sequence(variant: str) -> list[str]:
    return INVESTIGATOR_SEQUENCE if variant == "investigator" else BASE_SEQUENCE


def create_job(db, user: dict, query: str, raw_variant: str) -> dict:
    variant = _variant(raw_variant)
    job_id = uuid.uuid4().hex[:12]
    lo, hi = ETA_MINUTES[variant]
    doc = {
        "job_id": job_id,
        "user_id": user["id"],
        "email": user.get("email") or "",
        "query": query,
        "variant": variant,
        "status": "queued",
        "percent": 0,
        "phase": "Queued",
        "attempts": 0,
        "submitted_at": now_iso(),
        "started_at": None,
        "heartbeat_at": now_iso(),
        "finished_at": None,
        "elapsed_seconds": None,
        "eta_low_minutes": lo,
        "eta_high_minutes": hi,
        "dossier": None,
        "sources": None,
        "error": None,
    }
    _col(db).document(job_id).set(doc)
    return doc


def _update(db, job_id: str, payload: dict) -> None:
    try:
        _col(db).document(job_id).set(payload, merge=True)
    except Exception as exc:
        logger.warning("job %s update failed: %s", job_id, exc)


def submit(db, job_id: str, query: str, variant: str) -> None:
    task = asyncio.create_task(_guarded_run(db, job_id, query, variant))
    _tasks[job_id] = task
    task.add_done_callback(lambda _t: _tasks.pop(job_id, None))


async def _guarded_run(db, job_id: str, query: str, variant: str) -> None:
    async with _sem():
        await _run(db, job_id, query, variant)


async def _run(db, job_id: str, query: str, variant: str) -> None:
    run_root = Path(tempfile.gettempdir()) / "mensia-vin" / job_id
    started = time.time()
    _update(db, job_id, {"status": "running", "started_at": now_iso(),
                         "heartbeat_at": now_iso()})

    heartbeat = asyncio.create_task(_keep_alive(db, job_id))
    seq = _sequence(variant)
    done: list[str] = []

    def on_stage(name: str) -> None:
        if name in seq:
            done.append(name)
        percent = min(99, int(100 * len(done) / len(seq)))
        _update(db, job_id, {"percent": percent,
                             "phase": PHASE_LABELS.get(name, "Working"),
                             "heartbeat_at": now_iso()})

    try:
        run_root.mkdir(parents=True, exist_ok=True)
        path = await vin.run_pipeline(
            query=query,
            out_dir=run_root,
            skip_redteam=(variant != "investigator"),
            skip_polish=False,
            on_stage=on_stage,
        )
        dossier_text = Path(path).read_text(encoding="utf-8")
        sources = _read_sources(Path(path).parent)
        _update(db, job_id, {
            "status": "complete",
            "percent": 100,
            "phase": "Done",
            "finished_at": now_iso(),
            "elapsed_seconds": round(time.time() - started),
            "dossier": dossier_text[:400_000],
            "sources": sources,
            "error": None,
        })
        print(f"[vin] job {job_id} complete in {round(time.time() - started)}s")
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        logger.warning("vin job %s failed: %s", job_id, reason[:300])
        _update(db, job_id, {
            "status": "failed",
            "phase": "Failed",
            "finished_at": now_iso(),
            "elapsed_seconds": round(time.time() - started),
            "error": reason[:600],
        })
        _refund(db, job_id)
    finally:
        heartbeat.cancel()
        shutil.rmtree(run_root, ignore_errors=True)


def _read_sources(run_dir: Path) -> list[dict]:
    src_file = run_dir / "sources.json"
    if not src_file.exists():
        return []
    try:
        raw = json.loads(src_file.read_text(encoding="utf-8"))
    except Exception:
        return []
    out = []
    for item in raw if isinstance(raw, list) else []:
        out.append({
            "id": item.get("source_id") or item.get("id") or "",
            "title": (item.get("title") or "")[:300],
            "url": item.get("url") or "",
            "date": (item.get("date") or "")[:40],
        })
    return out[:80]


async def _keep_alive(db, job_id: str) -> None:
    while True:
        await asyncio.sleep(30)
        _update(db, job_id, {"heartbeat_at": now_iso()})


def _refund(db, job_id: str) -> None:
    job = _col(db).document(job_id).get()
    if not job.exists:
        return
    doc = job.to_dict()
    if doc.get("refunded") or doc.get("variant") is None:
        return
    quotas.refund(db, doc["user_id"], "vin", doc.get("variant"))
    _update(db, job_id, {"refunded": True})


def public_view(doc: dict) -> dict:
    """What the browser is allowed to see. No stage names, no internals."""
    return {
        "job_id": doc.get("job_id"),
        "query": doc.get("query"),
        "variant": doc.get("variant"),
        "status": doc.get("status"),
        "percent": doc.get("percent", 0),
        "phase": doc.get("phase"),
        "eta_low_minutes": doc.get("eta_low_minutes"),
        "eta_high_minutes": doc.get("eta_high_minutes"),
        "elapsed_seconds": doc.get("elapsed_seconds"),
        "submitted_at": doc.get("submitted_at"),
        "finished_at": doc.get("finished_at"),
        "dossier": doc.get("dossier"),
        "sources": doc.get("sources"),
        "error": "That run hit a wall on the research side. Your query was given back."
                 if doc.get("status") == "failed" else None,
    }


def load_mine(db, job_id: str, user_id: str) -> dict | None:
    snap = _col(db).document(job_id).get()
    if not snap.exists or snap.to_dict().get("user_id") != user_id:
        return None
    return public_view(snap.to_dict())


def list_mine(db, user_id: str, limit: int = 20) -> list[dict]:
    out = []
    for snap in _col(db).where(field_path="user_id", op_string="==", value=user_id).limit(200).stream():
        out.append(public_view(snap.to_dict()))
    out.sort(key=lambda d: d.get("submitted_at") or "", reverse=True)
    return out[:limit]


async def recover_orphans(db) -> int:
    """Re-queue runs left mid-flight by a redeploy. They restart from the beginning;
    there is no stage-level resume, and a second attempt is the limit."""
    if db is None:
        return 0
    revived = 0
    for status in ("running", "queued"):
        try:
            snaps = list(_col(db).where(field_path="status", op_string="==", value=status).limit(50).stream())
        except Exception as exc:
            logger.warning("orphan scan (%s) failed: %s", status, exc)
            continue
        for snap in snaps:
            doc = snap.to_dict()
            if doc.get("job_id") in _tasks:
                continue
            attempts = int(doc.get("attempts", 0))
            stale = _is_stale(doc)
            if status == "queued" or stale:
                if attempts >= MAX_ATTEMPTS:
                    _update(db, snap.id, {"status": "failed", "phase": "Failed",
                                          "error": "run was interrupted too many times",
                                          "finished_at": now_iso()})
                    _refund(db, snap.id)
                    continue
                _update(db, snap.id, {"status": "queued", "percent": 0,
                                      "phase": "Re-queued after a restart",
                                      "attempts": attempts + 1,
                                      "heartbeat_at": now_iso()})
                submit(db, snap.id, doc["query"], doc.get("variant", "base"))
                revived += 1
    if revived:
        logger.info("re-queued %d interrupted Vin run(s)", revived)
    return revived


def _is_stale(doc: dict) -> bool:
    beat = doc.get("heartbeat_at") or doc.get("submitted_at")
    try:
        then = datetime.fromisoformat(beat)
    except Exception:
        return True
    age = (datetime.now(timezone.utc) - then).total_seconds()
    return age > HEARTBEAT_STALE_S
