"""
Forwardable, tamper-evident case bundle — the Track-3 "reporting / response" path.

Packages a case into a self-contained JSON evidence bundle a bank or fraud team can ingest
and verify: the verdict + recommendation, the SHA-256 of the analyzed media (chain of
custody), a bundle hash, and an optional HMAC signature (REPORT_SIGNING_KEY) so the receiver
can confirm the bundle was not altered in transit. The media itself is never included.
"""
import hashlib
import hmac
import json
from datetime import datetime, timezone
from typing import Any, Dict, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import REPORT_SIGNING_KEY, FRAUD_WEBHOOK_URL


def build_bundle(case: Dict[str, Any]) -> Dict[str, Any]:
    """Build a signed (if a key is set) evidence bundle from a case ledger row."""
    env = json.loads(case.get("report_json") or "{}")
    core = {
        "schema": "truthy.case-bundle/v1",
        "caseId": case.get("case_id"),
        "createdAt": case.get("created_at"),
        "mediaType": case.get("media_type"),
        "mediaSha256": case.get("sha256"),
        "verdict": case.get("verdict"),
        "riskLevel": case.get("risk_level"),
        "confidence": case.get("confidence"),
        "aiProbability": case.get("ai_probability"),
        "modelBacked": bool(case.get("model_backed")),
        "calibrated": bool(case.get("calibrated")),
        "reviewStatus": case.get("review_status"),
        "reviewerDecision": case.get("reviewer_decision"),
        "recommendation": env.get("recommendation"),
        "signals": env.get("signals", []),
        "provenance": env.get("provenance", {}),
        "generatedAt": datetime.now(timezone.utc).isoformat(),
    }
    canonical = json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
    bundle_sha = hashlib.sha256(canonical).hexdigest()
    integrity = {"bundleSha256": bundle_sha, "signed": False}
    if REPORT_SIGNING_KEY:
        integrity.update({
            "signed": True, "algo": "HMAC-SHA256",
            "signature": hmac.new(REPORT_SIGNING_KEY.encode(), canonical, hashlib.sha256).hexdigest(),
            "verify": "HMAC-SHA256 over the canonical JSON of `case` (sorted keys), keyed by the shared REPORT_SIGNING_KEY.",
        })
    else:
        integrity["note"] = "Unsigned. Set REPORT_SIGNING_KEY to make forwarded bundles tamper-evident."
    return {"case": core, "integrity": integrity}


def verify_bundle(bundle: Dict[str, Any], key: str) -> bool:
    """Receiver-side integrity check (also used by the self-test)."""
    core = bundle.get("case", {})
    sig = bundle.get("integrity", {}).get("signature")
    if not sig or not key:
        return False
    canonical = json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
    expected = hmac.new(key.encode(), canonical, hashlib.sha256).hexdigest()
    return hmac.compare_digest(sig, expected)


def forward_bundle(bundle: Dict[str, Any], timeout: float = 8.0) -> Tuple[bool, str]:
    """POST the bundle to the configured fraud webhook. Returns (delivered, report_ref).
    With no webhook configured the case is still recorded as reported (delivered=False)."""
    ref = "RPT-" + str(bundle["case"].get("caseId")) + "-" + bundle["integrity"]["bundleSha256"][:8]
    if not FRAUD_WEBHOOK_URL:
        return False, ref
    try:
        import requests
        r = requests.post(FRAUD_WEBHOOK_URL, json=bundle, timeout=timeout)
        return (200 <= r.status_code < 300), ref
    except Exception:
        return False, ref


if __name__ == "__main__":
    # ponytail: self-check — signed bundle verifies; any tamper fails.
    import config
    config.REPORT_SIGNING_KEY = "testkey"; globals()["REPORT_SIGNING_KEY"] = "testkey"
    case = {"case_id": "TRV-1", "created_at": "t", "media_type": "image", "sha256": "abc",
            "verdict": "AI_GENERATED", "risk_level": "high", "confidence": "High",
            "ai_probability": 0.9, "model_backed": 1, "calibrated": 1, "review_status": "queued",
            "report_json": json.dumps({"recommendation": "Do not pay.", "signals": [], "provenance": {}})}
    b = build_bundle(case)
    assert b["integrity"]["signed"] and verify_bundle(b, "testkey")
    tampered = json.loads(json.dumps(b)); tampered["case"]["verdict"] = "HUMAN"
    assert not verify_bundle(tampered, "testkey"), "tamper must fail"
    print("bundle self-check OK", b["integrity"]["bundleSha256"][:12])
