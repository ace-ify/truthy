"""
Minimal case ledger for evidence & governance (Track 3).

Stores verdict metadata + a SHA-256 of the analyzed media — NEVER the media itself.
This keeps the transient-processing privacy posture (files are deleted after analysis)
while still giving every check a durable, auditable record with a chain-of-custody hash,
a human-review status, and an escalation path.

Stdlib only: sqlite3 + hashlib. WAL mode + a module-level lock handle the low hackathon
write volume. ponytail: global lock is fine here; move to per-connection if throughput matters.
"""
import hashlib
import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import CASES_DB_PATH

_lock = threading.Lock()
_initialized = False

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    case_id       TEXT PRIMARY KEY,
    created_at    TEXT NOT NULL,
    media_type    TEXT NOT NULL,
    filename      TEXT,
    size_bytes    INTEGER,
    sha256        TEXT NOT NULL,
    verdict       TEXT NOT NULL,
    risk_level    TEXT NOT NULL,
    confidence    TEXT,
    ai_probability REAL,
    calibrated    INTEGER DEFAULT 0,
    model_backed  INTEGER DEFAULT 0,
    report_json   TEXT NOT NULL,
    review_status TEXT DEFAULT 'none',   -- none | queued | reviewed
    reviewer_note TEXT,
    reviewer_decision TEXT,              -- confirm | override_authentic | override_fake
    reviewed_at   TEXT,
    reported_at   TEXT,                  -- when forwarded to a fraud/response channel
    report_ref    TEXT                   -- reference id of the forwarded report
);
CREATE INDEX IF NOT EXISTS idx_cases_review ON cases(review_status);
CREATE INDEX IF NOT EXISTS idx_cases_created ON cases(created_at);
"""

# Columns added after the original schema shipped; added to existing DBs on init.
_MIGRATIONS = [
    ("reported_at", "ALTER TABLE cases ADD COLUMN reported_at TEXT"),
    ("report_ref", "ALTER TABLE cases ADD COLUMN report_ref TEXT"),
]


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(CASES_DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def _ensure_init() -> None:
    # Idempotent and cheap: run every call so an in-memory flag can't desync from the
    # actual DB file (e.g. if the file is recreated at runtime). CREATE ... IF NOT EXISTS
    # + guarded ALTERs are near-free at hackathon volume.
    with _lock:
        conn = _connect()
        try:
            conn.executescript(_SCHEMA)
            existing = {r[1] for r in conn.execute("PRAGMA table_info(cases)").fetchall()}
            for col, ddl in _MIGRATIONS:
                if col not in existing:
                    conn.execute(ddl)
            conn.commit()
        finally:
            conn.close()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _short_id() -> str:
    # Human-friendly case id, e.g. TRV-3F9A2C7B
    return "TRV-" + uuid.uuid4().hex[:8].upper()


def record_case(envelope: Dict[str, Any], media_bytes: bytes) -> Dict[str, Any]:
    """Persist a verdict envelope + media hash. Returns the case_id/sha256/createdAt.

    Auto-queues for human review when the envelope asks to escalate.
    """
    _ensure_init()
    case_id = _short_id()
    created_at = datetime.now(timezone.utc).isoformat()
    sha = sha256_bytes(media_bytes)
    review_status = "queued" if envelope.get("escalate") else "none"

    row = {
        "case_id": case_id,
        "created_at": created_at,
        "media_type": envelope.get("mediaType", "unknown"),
        "filename": envelope.get("filename"),
        "size_bytes": len(media_bytes),
        "sha256": sha,
        "verdict": envelope.get("verdict", "INCONCLUSIVE"),
        "risk_level": envelope.get("riskLevel", "medium"),
        "confidence": envelope.get("confidence", "Low"),
        "ai_probability": envelope.get("aiProbability", 0.5),
        "calibrated": 1 if envelope.get("calibrated") else 0,
        "model_backed": 1 if envelope.get("modelBacked") else 0,
        "report_json": json.dumps(envelope),
        "review_status": review_status,
    }
    with _lock:
        conn = _connect()
        try:
            conn.execute(
                """INSERT INTO cases (case_id, created_at, media_type, filename, size_bytes,
                    sha256, verdict, risk_level, confidence, ai_probability, calibrated,
                    model_backed, report_json, review_status)
                   VALUES (:case_id, :created_at, :media_type, :filename, :size_bytes,
                    :sha256, :verdict, :risk_level, :confidence, :ai_probability, :calibrated,
                    :model_backed, :report_json, :review_status)""",
                row,
            )
            conn.commit()
        finally:
            conn.close()
    return {"caseId": case_id, "sha256": sha, "createdAt": created_at, "reviewStatus": review_status}


def get_case(case_id: str) -> Optional[Dict[str, Any]]:
    _ensure_init()
    conn = _connect()
    try:
        cur = conn.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,))
        r = cur.fetchone()
        return dict(r) if r else None
    finally:
        conn.close()


def list_cases(review_status: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    _ensure_init()
    conn = _connect()
    try:
        if review_status:
            cur = conn.execute(
                "SELECT case_id, created_at, media_type, filename, verdict, risk_level, "
                "confidence, review_status FROM cases WHERE review_status = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (review_status, limit),
            )
        else:
            cur = conn.execute(
                "SELECT case_id, created_at, media_type, filename, verdict, risk_level, "
                "confidence, review_status FROM cases ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def escalate_case(case_id: str) -> bool:
    """Mark a case as queued for human review."""
    _ensure_init()
    with _lock:
        conn = _connect()
        try:
            cur = conn.execute(
                "UPDATE cases SET review_status='queued' WHERE case_id=? AND review_status='none'",
                (case_id,),
            )
            conn.commit()
            # queued->queued or already reviewed still counts as "exists"
            return cur.rowcount > 0 or get_case(case_id) is not None
        finally:
            conn.close()


def review_case(case_id: str, decision: str, note: str = "") -> bool:
    """Record a human reviewer's decision. decision in {confirm, override_authentic, override_fake}."""
    _ensure_init()
    reviewed_at = datetime.now(timezone.utc).isoformat()
    with _lock:
        conn = _connect()
        try:
            cur = conn.execute(
                "UPDATE cases SET review_status='reviewed', reviewer_decision=?, "
                "reviewer_note=?, reviewed_at=? WHERE case_id=?",
                (decision, note, reviewed_at, case_id),
            )
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()


def mark_reported(case_id: str, report_ref: str) -> bool:
    """Record that a case was forwarded to a fraud/response channel."""
    _ensure_init()
    reported_at = datetime.now(timezone.utc).isoformat()
    with _lock:
        conn = _connect()
        try:
            cur = conn.execute(
                "UPDATE cases SET reported_at=?, report_ref=? WHERE case_id=?",
                (reported_at, report_ref, case_id),
            )
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()


if __name__ == "__main__":
    # ponytail: self-check — record -> fetch -> review -> report round-trips.
    import tempfile, os
    import config
    config.CASES_DB_PATH = os.path.join(tempfile.gettempdir(), "cases_selfcheck.db")
    globals()["CASES_DB_PATH"] = config.CASES_DB_PATH
    if os.path.exists(config.CASES_DB_PATH):
        os.unlink(config.CASES_DB_PATH)
    _initialized = False
    env = {"mediaType": "image", "verdict": "AI_GENERATED", "riskLevel": "high",
           "confidence": "High", "aiProbability": 0.9, "escalate": True, "filename": "x.jpg"}
    meta = record_case(env, b"hello-bytes")
    assert meta["caseId"].startswith("TRV-")
    c = get_case(meta["caseId"])
    assert c["verdict"] == "AI_GENERATED" and c["review_status"] == "queued"
    assert c["sha256"] == sha256_bytes(b"hello-bytes")
    assert review_case(meta["caseId"], "confirm", "looks fake")
    assert get_case(meta["caseId"])["review_status"] == "reviewed"
    assert mark_reported(meta["caseId"], "RPT-TEST")
    assert get_case(meta["caseId"])["report_ref"] == "RPT-TEST"
    assert len(list_cases()) >= 1
    os.unlink(config.CASES_DB_PATH)
    print("governance store self-check OK:", meta["caseId"])
