"""
Provenance aggregation — the Track-3 differentiator.

Combines three provenance signals, in order of trust:
  1. C2PA / Content Credentials  (cryptographically verifiable — strongest)
  2. EXIF via piexif             (camera make/model/date/GPS/lens, or AI software marker)
  3. JPEG quantization tables    (recompression context, e.g. messaging-app re-encode)

Precedence rule: a VERIFIED C2PA manifest, or a coherent camera-capture EXIF chain, is
stronger evidence of origin than any pixel heuristic and can set the headline verdict. An
unverified marker or forgeable EXIF only nudges / fills an Inconclusive gap — never
overrides a confident model.
"""
import logging
from typing import Any, Dict, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from core.provenance.c2pa_verify import verify_c2pa, claim_generator_is_ai

logger = logging.getLogger(__name__)

_AI_SOFTWARE = ("stable diffusion", "midjourney", "dall-e", "dalle", "comfyui",
                "automatic1111", "invokeai", "invoke ai", "novelai", "firefly", "flux",
                "ideogram", "leonardo", "runway", "playground", "bing image creator",
                "gemini", "imagen", "openai")


def _b2s(v) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, bytes):
        return v.decode("ascii", "ignore").strip("\x00 ").strip() or None
    return str(v).strip() or None


def _read_exif(raw_bytes: bytes) -> Dict[str, Any]:
    out = {"make": None, "model": None, "software": None, "datetime": None,
           "lens": None, "has_gps": False, "present": False}
    try:
        import piexif
        exif = piexif.load(raw_bytes)
    except Exception:  # noqa: BLE001 — non-JPEG/webp or no exif
        return out
    zeroth = exif.get("0th", {}) or {}
    exsub = exif.get("Exif", {}) or {}
    gps = exif.get("GPS", {}) or {}
    try:
        out["make"] = _b2s(zeroth.get(piexif.ImageIFD.Make))
        out["model"] = _b2s(zeroth.get(piexif.ImageIFD.Model))
        out["software"] = _b2s(zeroth.get(piexif.ImageIFD.Software))
        out["datetime"] = _b2s(zeroth.get(piexif.ImageIFD.DateTime))
        out["lens"] = _b2s(exsub.get(piexif.ExifIFD.LensModel))
        out["has_gps"] = bool(gps)
        out["present"] = any([out["make"], out["model"], out["software"], out["datetime"]])
    except Exception:  # noqa: BLE001
        pass
    return out


def _quant_summary(pil_image) -> Dict[str, Any]:
    """JPEG quantization-table context (recompression heuristic, not an AI signal)."""
    q = getattr(pil_image, "quantization", None)
    if not q:
        return {"has_tables": False, "note": "No JPEG quantization tables (not a JPEG / re-saved)."}
    try:
        table0 = list(q.values())[0]
        avg = sum(table0) / len(table0)
        # Higher avg quantization step => heavier compression / lower quality.
        if avg > 20:
            note = "Heavy JPEG recompression (typical of messaging-app forwarding)."
        elif avg > 8:
            note = "Moderate JPEG compression."
        else:
            note = "Light JPEG compression (near-original quality)."
        return {"has_tables": True, "avg_quant": round(avg, 2), "n_tables": len(q), "note": note}
    except Exception:  # noqa: BLE001
        return {"has_tables": True, "note": "Quantization tables present."}


def _software_is_ai(software: Optional[str]) -> bool:
    if not software:
        return False
    s = software.lower()
    return any(m in s for m in _AI_SOFTWARE)


def analyze_provenance(raw_bytes: bytes, pil_image=None, original_format: str = "JPEG") -> Dict[str, Any]:
    """Return the provenance panel dict + a flag/strength/score for the precedence rule."""
    c2pa = verify_c2pa(raw_bytes=raw_bytes, original_format=original_format)
    exif = _read_exif(raw_bytes)
    quant = _quant_summary(pil_image) if pil_image is not None else {"has_tables": False}

    flag, strength, score, summary = "none", "none", None, "No usable provenance found."

    if c2pa.get("verified"):
        if claim_generator_is_ai(c2pa.get("claim_generator")):
            flag, strength, score = "ai", "strong", 0.95
            summary = f"Verified Content Credential says this was made with {c2pa['claim_generator']}."
        else:
            flag, strength, score = "authentic", "strong", 0.05
            summary = (f"Verified Content Credential from {c2pa.get('claim_generator') or 'a signer'} "
                       "— origin cryptographically attested.")
    elif _software_is_ai(exif.get("software")):
        flag, strength, score = "ai", "moderate", 0.75
        summary = f"EXIF software tag names an AI generator: {exif['software']} (forgeable, so moderate)."
    elif exif.get("make") and exif.get("model") and exif.get("datetime"):
        flag, strength, score = "authentic", "moderate", 0.30
        summary = (f"Camera-capture EXIF chain present ({exif['make']} {exif['model']}, "
                   f"{exif['datetime']}). Consistent with a real photo (EXIF is forgeable, so moderate).")
    elif c2pa.get("present"):
        flag, strength, score = "unknown", "weak", None
        summary = "A Content Credential is present but could not be cryptographically verified."

    return {
        "flag": flag,            # ai | authentic | unknown | none
        "strength": strength,    # strong | moderate | weak | none
        "score": score,          # AI-leaning 0..1 (None if uninformative)
        "summary": summary,
        "c2pa": c2pa,
        "exif": exif,
        "quant": quant,
    }


def apply_precedence(verdict, prov: Dict[str, Any]) -> None:
    """Mutate an ImageVerdict in place per the precedence rule. Only STRONG provenance
    overrides the model; MODERATE only fills an Inconclusive gap."""
    strength = prov.get("strength")
    flag = prov.get("flag")
    if strength == "strong" and flag in ("ai", "authentic"):
        if flag == "ai":
            verdict.verdict = "AI Generated"
            verdict.overall_ai_probability = max(verdict.overall_ai_probability, 0.95)
        else:
            verdict.verdict = "Human Created"
            verdict.overall_ai_probability = min(verdict.overall_ai_probability, 0.05)
        verdict.confidence = "High"
        verdict.escalate = False
        verdict.explanation = prov["summary"] + " " + verdict.explanation
    elif strength == "moderate" and verdict.verdict == "Inconclusive":
        if flag == "ai":
            verdict.verdict, verdict.overall_ai_probability = "AI Generated", 0.7
        elif flag == "authentic":
            verdict.verdict, verdict.overall_ai_probability = "Human Created", 0.3
        verdict.confidence = "Low"  # moderate provenance is forgeable — stay humble
        verdict.escalate = True
        verdict.explanation = prov["summary"] + " " + verdict.explanation


if __name__ == "__main__":
    # ponytail: self-check — clean bytes yield no provenance; AI software EXIF flags ai.
    r = analyze_provenance(b"\xff\xd8\xff\xe0 not a real jpeg", None)
    assert r["flag"] in ("none", "unknown")
    assert not _software_is_ai("Adobe Lightroom")
    assert _software_is_ai("Stable Diffusion WebUI")

    class _V:
        verdict = "Inconclusive"; overall_ai_probability = 0.5; confidence = "Low"
        escalate = True; explanation = "base."
    v = _V()
    apply_precedence(v, {"strength": "strong", "flag": "ai", "score": 0.95, "summary": "Verified AI."})
    assert v.verdict == "AI Generated" and v.confidence == "High"
    print("provenance self-check OK")
