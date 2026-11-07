"""Firestore-backed usage limits, access codes, and per-user grant state for Mensia.

Everything here fails OPEN: if Firestore is unreachable the request is allowed and
nothing is recorded, so a database blip never takes the site down.
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no I/O/0/1 so codes survive being emailed

FREE_LIMITS = {"search": 5, "deep research": 1, "vin": 1}
EXTRA_WEEKLY_LIMITS = {"search": 10, "deep research": 3, "vin_base": 2, "vin_investigator": 2}

USERS = "mensia_users"
CODES = "mensia_access_codes"

BUCKETS = ("search", "deep research", "vin_base", "vin_investigator")


def _db():
    try:
        import firebase_admin
        if not firebase_admin._apps:
            return None
        from firebase_admin import firestore
        return firestore.client()
    except Exception as exc:
        logger.warning("Firestore unavailable: %s", exc)
        return None


def db_client():
    return _db()


def firestore_transactional():
    from firebase_admin import firestore as _fs
    return _fs.transactional


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def week_key(at: datetime | None = None) -> str:
    year, week, _ = (at or now_utc()).isocalendar()
    return f"{year}-W{week:02d}"


def code_hash(code: str) -> str:
    normalised = "".join((code or "").upper().split())
    pepper = os.getenv("MENSIA_CODE_PEPPER", "mensia-access-v1")
    return hashlib.sha256(f"{pepper}:{normalised}".encode()).hexdigest()


def generate_code(kind: str = "extra") -> tuple[str, str]:
    body = "-".join("".join(secrets.choice(ALPHABET) for _ in range(4)) for _ in range(3))
    code = f"{'MES' if kind == 'extra' else 'MSA'}-{body}"
    return code, code_hash(code)


def _blank_usage(user_id: str, email: str | None) -> dict:
    return {
        "user_id": user_id,
        "email": email or "",
        "plan": "free",
        "lifetime": {"search": 0, "deep research": 0, "vin": 0},
        "weekly": {"key": week_key(), "search": 0, "deep research": 0,
                   "vin_base": 0, "vin_investigator": 0},
        "grant": None,
        "created_at": now_utc().isoformat(),
        "updated_at": now_utc().isoformat(),
    }


def _fresh_weekly() -> dict:
    return {"key": week_key(), "search": 0, "deep research": 0,
            "vin_base": 0, "vin_investigator": 0}


def ensure_weekly(doc: dict) -> dict:
    if (doc.get("weekly") or {}).get("key") != week_key():
        doc["weekly"] = _fresh_weekly()
    return doc


def bucket_for(mode: str, variant: str | None = None) -> str:
    if mode == "vin":
        return "vin_investigator" if (variant or "base").lower().startswith("inv") else "vin_base"
    return mode


def effective_limits(doc: dict | None) -> dict:
    """special -> unlimited, extra -> weekly caps, free -> lifetime caps."""
    plan = (doc or {}).get("plan", "free")
    if plan == "special":
        return {"plan": "special", "scope": None, "used": None}
    if plan == "extra":
        weekly = (doc or {}).get("weekly") or {}
        return {"plan": "extra", "scope": EXTRA_WEEKLY_LIMITS,
                "used": {b: int(weekly.get(b, 0)) for b in BUCKETS}}
    lifetime = (doc or {}).get("lifetime") or {}
    vin_used = int(lifetime.get("vin", 0))
    return {"plan": "free", "scope": {**FREE_LIMITS},
            "used": {"search": int(lifetime.get("search", 0)),
                     "deep research": int(lifetime.get("deep research", 0)),
                     "vin_base": vin_used, "vin_investigator": vin_used}}


def _cap_used(res: dict, bucket: str) -> tuple[int | None, int]:
    if res["plan"] == "special":
        return None, 0
    if res["plan"] == "free" and bucket.startswith("vin"):
        return FREE_LIMITS["vin"], res["used"][bucket]
    return res["scope"].get(bucket), res["used"].get(bucket, 0)


def snapshot(doc: dict | None) -> dict:
    """Quota picture handed to the frontend by /limits."""
    res = effective_limits(doc)
    rows = {}
    for bucket in BUCKETS:
        cap, used = _cap_used(res, bucket)
        rows[bucket] = {
            "used": used,
            "cap": cap,
            "remaining": None if cap is None else max(cap - used, 0),
            "exhausted": cap is not None and used >= cap,
        }
    window = ("unlimited" if res["plan"] == "special"
              else "weekly" if res["plan"] == "extra" else "lifetime")
    # The unlimited tier is never named to the client; it just stops having a ceiling.
    return {"plan": "unlimited" if res["plan"] == "special" else res["plan"],
            "window": window, "week": week_key(), "rows": rows}


def get_or_create(db, user_id: str, email: str | None) -> dict:
    ref = db.collection(USERS).document(user_id)
    snap = ref.get()
    doc = snap.to_dict() if snap.exists else _blank_usage(user_id, email)
    doc = ensure_weekly(doc)
    if not snap.exists:
        ref.set(doc)
    return doc


def read_user(db, user_id: str) -> dict | None:
    snap = db.collection(USERS).document(user_id).get()
    if not snap.exists:
        return None
    return ensure_weekly(snap.to_dict())


def check_and_consume(db, user_id: str, email: str | None, mode: str,
                      variant: str | None = None) -> tuple[bool, str, dict]:
    """Atomically charge one query against the account's allowance."""
    bucket = bucket_for(mode, variant)
    ref = db.collection(USERS).document(user_id)
    outcome: dict = {}

    @firestore_transactional()
    def _charge(tx):
        snap = ref.get(transaction=tx)
        doc = snap.to_dict() if snap.exists else _blank_usage(user_id, email)
        doc = ensure_weekly(doc)
        res = effective_limits(doc)
        cap, used = _cap_used(res, bucket)
        if cap is not None and used >= cap:
            outcome.update(allowed=False, plan=res["plan"])
            return None
        if res["plan"] == "extra":
            doc["weekly"][bucket] = used + 1
        elif res["plan"] == "free":
            key = "vin" if bucket.startswith("vin") else bucket
            doc["lifetime"][key] = int((doc.get("lifetime") or {}).get(key, 0)) + 1
        doc["updated_at"] = now_utc().isoformat()
        tx.set(ref, doc, merge=True)
        outcome.update(allowed=True, plan=res["plan"])
        return doc

    try:
        _charge(db.transaction())
    except Exception as exc:
        logger.warning("quota transaction failed for %s: %s", user_id, exc)
        return True, "", snapshot(None)   # fail open

    after = read_user(db, user_id)
    if outcome.get("allowed"):
        return True, "", snapshot(after)

    res = effective_limits(after)
    cap, _used = _cap_used(res, bucket)
    if bucket.startswith("vin"):
        msg = ("You've used your Vin run. Vin is a single-run lifetime allowance "
               "on the free tier — sign in to a fresh account won't help, but extra "
               "access is on the Increased Access page.")
    else:
        msg = (f"That's all {cap} of your '{bucket}' queries "
               f"({snapshot(after)['window']} allowance).")
    return False, msg, snapshot(after)


def refund(db, user_id: str, mode: str, variant: str | None = None) -> None:
    """Give a query back when a run failed through no fault of the user."""
    bucket = bucket_for(mode, variant)
    ref = db.collection(USERS).document(user_id)

    @firestore_transactional()
    def _give_back(tx):
        snap = ref.get(transaction=tx)
        if not snap.exists:
            return None
        doc = ensure_weekly(snap.to_dict())
        plan = doc.get("plan", "free")
        if plan == "free":
            key = "vin" if bucket.startswith("vin") else bucket
            life = doc.get("lifetime") or {}
            life[key] = max(int(life.get(key, 0)) - 1, 0)
            doc["lifetime"] = life
        elif plan == "extra":
            doc["weekly"][bucket] = max(int(doc["weekly"].get(bucket, 0)) - 1, 0)
        doc["updated_at"] = now_utc().isoformat()
        tx.set(ref, doc, merge=True)
        return doc

    try:
        _give_back(db.transaction())
    except Exception as exc:
        logger.warning("quota refund failed for %s: %s", user_id, exc)


def redeem_code(db, user_id: str, email: str | None, raw_code: str) -> tuple[bool, str, dict]:
    """Single-use code, permanently bound to the account that claimed it.

    'special' codes unlock the hidden unlimited tier; the response never says that
    word, it simply reports the plan the account is now on.
    """
    code = "".join((raw_code or "").upper().split())
    if len(code) < 8:
        return False, "That code doesn't look right. Check for a typo and try again.", {}

    cref = db.collection(CODES).document(code_hash(code))
    csnap = cref.get()
    if not csnap.exists:
        return False, "That code doesn't look right. Check for a typo and try again.", {}
    rec = csnap.to_dict()
    if rec.get("status") == "used" and rec.get("used_by") != user_id:
        return False, "This code has already been claimed.", {}

    kind = rec.get("kind", "extra")
    uref = db.collection(USERS).document(user_id)
    usnap = uref.get()
    doc = usnap.to_dict() if usnap.exists else _blank_usage(user_id, email)

    doc["plan"] = "special" if kind == "special" else "extra"
    doc["weekly"] = _fresh_weekly()
    doc["email"] = email or doc.get("email", "")
    doc["grant"] = {"kind": kind, "claimed_at": now_utc().isoformat()}
    doc["updated_at"] = now_utc().isoformat()
    uref.set(doc, merge=True)
    cref.set({"status": "used", "used_by": user_id, "used_email": email or "",
              "used_at": now_utc().isoformat()}, merge=True)
    return True, "Access unlocked. Your limits have been raised.", snapshot(doc)


def store_code(db, kind: str, code: str, hashed: str) -> None:
    db.collection(CODES).document(hashed).set({
        "kind": kind,
        "hint": code[:7],
        "status": "unused",
        "created_at": now_utc().isoformat(),
    })


def list_codes(db, limit: int = 200) -> list[dict]:
    out = []
    for snap in db.collection(CODES).limit(limit).stream():
        row = snap.to_dict()
        row["id"] = snap.id[:8]
        out.append(row)
    return out
