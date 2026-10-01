"""
C2PA / Content Credentials verification.

Reads and cryptographically verifies an embedded C2PA manifest (Content Credentials, as
attached by Adobe Firefly, Photoshop, some cameras, OpenAI, etc.) using the Rust-backed
`c2pa` library. A *verified* manifest is the strongest possible provenance evidence — far
stronger than any pixel heuristic — because it is a signed statement of origin/edits.

Falls back to a raw byte-scan for the C2PA/JUMBF marker when the library is unavailable or
errors, reporting "present but unverified" — we never claim verification we didn't perform.
"""
import io
import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_MIME = {
    "JPEG": "image/jpeg", "JPG": "image/jpeg", "PNG": "image/png",
    "WEBP": "image/webp", "TIFF": "image/tiff", "GIF": "image/gif",
    "MP4": "video/mp4", "MOV": "video/quicktime",
}

# AI generators known to write C2PA claim_generator strings.
_AI_GENERATORS = ("firefly", "dall-e", "dalle", "openai", "midjourney", "stable diffusion",
                  "gemini", "imagen", "adobe firefly", "bing", "designer")


def _empty(reason: str = "") -> Dict[str, Any]:
    return {
        "present": False, "signature_valid": None, "claim_generator": None,
        "validation_state": None, "verified": False, "note": reason,
    }


def _byte_scan(raw_bytes: Optional[bytes]) -> Dict[str, Any]:
    """Last-resort: detect a C2PA/JUMBF marker without verifying it."""
    if not raw_bytes:
        return _empty("no data")
    blob = raw_bytes[:5_000_000]
    for marker in (b"jumbf", b"c2pa", b"urn:c2pa", b"contentauth"):
        if marker in blob:
            return {
                "present": True, "signature_valid": None, "claim_generator": None,
                "validation_state": "unverified",
                "verified": False,
                "note": "Content Credential marker found but not cryptographically verified.",
            }
    return _empty("no content credential found")


def verify_c2pa(raw_bytes: Optional[bytes] = None, original_format: str = "JPEG",
                path: Optional[str] = None) -> Dict[str, Any]:
    """Return a provenance dict. Verified manifests set verified=True + signature_valid."""
    try:
        import c2pa  # noqa: F401
    except Exception:  # noqa: BLE001
        return _byte_scan(raw_bytes)

    reader = None
    try:
        import c2pa
        if path is not None:
            reader = c2pa.Reader(path)
        else:
            fmt = _MIME.get((original_format or "JPEG").upper(), "image/jpeg")
            reader = c2pa.Reader(fmt, stream=io.BytesIO(raw_bytes or b""))

        data = json.loads(reader.json())
        manifests = data.get("manifests", {}) or {}
        active_id = data.get("active_manifest")
        active = manifests.get(active_id, {}) if active_id else {}

        # claim generator name
        claim_gen = None
        cgi = active.get("claim_generator_info")
        if isinstance(cgi, list) and cgi:
            claim_gen = cgi[0].get("name")
        claim_gen = claim_gen or active.get("claim_generator")

        try:
            state = str(reader.get_validation_state())
        except Exception:  # noqa: BLE001
            state = "unknown"
        sig_valid = "valid" in state.lower() or "trusted" in state.lower()

        return {
            "present": True,
            "signature_valid": sig_valid,
            "verified": sig_valid,
            "claim_generator": claim_gen,
            "validation_state": state,
            "note": "C2PA manifest read and validated." if sig_valid
                    else "C2PA manifest present; signature not validated/trusted.",
        }
    except Exception as e:  # noqa: BLE001 — no manifest is the common case
        scan = _byte_scan(raw_bytes)
        if scan["present"]:
            return scan
        return _empty(f"no C2PA manifest ({str(e)[:60]})")
    finally:
        if reader is not None:
            try:
                reader.close()
            except Exception:  # noqa: BLE001
                pass


def claim_generator_is_ai(claim_generator: Optional[str]) -> bool:
    if not claim_generator:
        return False
    cg = claim_generator.lower()
    return any(g in cg for g in _AI_GENERATORS)


if __name__ == "__main__":
    # ponytail: self-check — clean bytes => not present; marker bytes => present-unverified.
    assert verify_c2pa(b"\xff\xd8\xff\xe0 plain jpeg bytes", "JPEG")["present"] is False
    assert _byte_scan(b"....jumbf....c2pa....")["present"] is True
    assert claim_generator_is_ai("Adobe Firefly") is True
    assert claim_generator_is_ai("Canon EOS") is False
    print("c2pa_verify self-check OK")
