"""
Unified verification report envelope.

All three modalities (voice / video / image) normalize into ONE report shape so the
frontend can render a single experience: verdict -> risk -> plain recommendation ->
evidence signals -> provenance -> actions. This module holds the normalization plus the
plain-language risk/recommendation layer required by Track 3 (a score is not advice).
"""
from typing import List, Optional

# Risk levels shown to the (non-technical) user.
RISK_HIGH = "high"
RISK_MEDIUM = "medium"
RISK_LOW = "low"

# Canonical verdict enum used across modalities.
V_AI = "AI_GENERATED"
V_HUMAN = "HUMAN"
V_INCONCLUSIVE = "INCONCLUSIVE"


def normalize_verdict(verdict: str) -> str:
    """Map the various internal verdict strings to the canonical enum."""
    v = (verdict or "").strip().lower()
    if v in ("ai generated", "ai_generated", "ai", "fake", "synthetic"):
        return V_AI
    if v in ("human created", "human", "human voice", "authentic", "real"):
        return V_HUMAN
    return V_INCONCLUSIVE


def headline(verdict: str) -> str:
    return {
        V_AI: "Likely fake / AI-generated",
        V_HUMAN: "Likely authentic",
        V_INCONCLUSIVE: "Needs review — not confirmed",
    }[verdict]


def risk_and_recommendation(verdict: str, confidence: str, escalate: bool) -> tuple[str, str]:
    """Map a verdict into a risk level and a plain, actionable, pay-context recommendation."""
    if verdict == V_AI:
        risk = RISK_HIGH
        rec = (
            "Treat this as likely fake. Do NOT act on any payment, transfer, or account "
            "instruction in it. Confirm with the person on a known, trusted number and report "
            "it to your bank or fraud team."
        )
    elif verdict == V_INCONCLUSIVE:
        risk = RISK_MEDIUM
        rec = (
            "We could not decide confidently. Do not rely on this message for a payment "
            "decision — send it to manual review and verify the request through a separate "
            "trusted channel first."
        )
    else:  # V_HUMAN
        risk = RISK_LOW
        rec = (
            "No strong signs of manipulation were found. For high-value or urgent money "
            "requests, still confirm through a separate trusted channel before paying."
        )
    if escalate and risk == RISK_LOW:
        risk = RISK_MEDIUM
    return risk, rec


def build_envelope(
    *,
    media_type: str,
    verdict: str,
    confidence: str,
    ai_probability: float,
    explanation: str,
    signals: Optional[List[dict]] = None,
    extras: Optional[dict] = None,
    provenance: Optional[dict] = None,
    calibrated: bool = False,
    model_backed: bool = False,
    escalate: bool = False,
    filename: Optional[str] = None,
) -> dict:
    """Produce the analytic part of a VerifyReport (case_id/sha256/timestamp added on persist)."""
    v = normalize_verdict(verdict)
    risk, recommendation = risk_and_recommendation(v, confidence, escalate)
    return {
        "mediaType": media_type,             # "voice" | "video" | "image"
        "verdict": v,                        # AI_GENERATED | HUMAN | INCONCLUSIVE
        "riskLevel": risk,                   # high | medium | low
        "confidence": confidence,            # High | Medium | Low
        "aiProbability": round(float(ai_probability), 3),
        "calibrated": bool(calibrated),
        "modelBacked": bool(model_backed),
        "escalate": bool(escalate),
        "headline": headline(v),
        "explanation": explanation,
        "recommendation": recommendation,
        "signals": signals or [],            # primary + supporting breakdown
        "provenance": provenance or {},      # C2PA / EXIF panel (image)
        "extras": extras or {},              # modality-specific (bpm, chunks, charts)
        "filename": filename,
    }


def envelope_from_image(verdict, filename: Optional[str] = None) -> dict:
    """ImageVerdict -> unified envelope."""
    return build_envelope(
        media_type="image",
        verdict=verdict.verdict,
        confidence=verdict.confidence,
        ai_probability=verdict.overall_ai_probability,
        explanation=verdict.explanation,
        signals=verdict.signal_breakdown,
        extras={
            "heatmapBase64": getattr(verdict, "heatmap_b64", None),
            "generatorGuess": getattr(verdict, "generator_guess", None),
        },
        provenance=getattr(verdict, "provenance", None),
        calibrated=getattr(verdict, "calibrated", False),
        model_backed=getattr(verdict, "model_backed", False),
        escalate=getattr(verdict, "escalate", False),
        filename=filename,
    )


def envelope_from_video(verdict, filename: Optional[str] = None) -> dict:
    """VideoVerdict -> unified envelope (voice = primary; rPPG/lip-sync = supporting)."""
    signals = []
    vr = getattr(verdict, "voice_results", {}) or {}
    if vr.get("evaluated"):
        signals.append({
            "analyzer": "voice_clone", "display_name": "Voice-clone model (wav2vec2)",
            "tier": "primary", "score": round(vr.get("voice_ai_prob", 0.5), 3),
            "confidence": 0.85, "weight": 2.5, "reasoning": vr.get("reasoning", ""),
        })
    rp = verdict.rppg_results or {}
    if rp.get("has_face"):
        signals.append({
            "analyzer": "rppg", "display_name": "Optical pulse rPPG (experimental)",
            "tier": "supporting", "score": round(rp.get("rppg_ai_score", 0.5), 3),
            "confidence": 0.4, "weight": 0.5, "reasoning": rp.get("reasoning", ""),
        })
    ls = verdict.lip_sync_results or {}
    if ls.get("lip_sync_evaluated"):
        signals.append({
            "analyzer": "lipsync", "display_name": "Lip-sync heuristic (experimental)",
            "tier": "supporting", "score": round(ls.get("kinematic_mismatch_score", 0.5), 3),
            "confidence": 0.4, "weight": 0.5, "reasoning": ls.get("reasoning", ""),
        })
    extras = {
        "fps": verdict.fps, "durationSeconds": verdict.duration_seconds,
        "totalFramesAnalyzed": verdict.total_frames,
        "hasBiologicalPulse": rp.get("has_pulse", False), "detectedBpm": rp.get("bpm"),
        "pulseSnr": rp.get("pulse_snr", 0.0),
        "biomechanicalViolations": ls.get("biomechanical_violations", 0),
        "bvpChartBase64": verdict.bvp_chart_b64,
    }
    return build_envelope(
        media_type="video", verdict=verdict.verdict, confidence=verdict.confidence,
        ai_probability=verdict.overall_ai_probability, explanation=verdict.explanation,
        signals=signals, extras=extras, model_backed=verdict.model_backed,
        escalate=verdict.escalate, filename=filename,
    )


def envelope_from_voice(result, language: str, explanation: str, filename: Optional[str] = None) -> dict:
    """Audio AnalysisResult -> unified envelope (wav2vec2 voice-clone = primary)."""
    p = float(result.overall_ai_probability)
    confidence = result.confidence if result.confidence in ("High", "Medium", "Low") else "Low"
    signals = [{
        "analyzer": "voice_clone", "display_name": "Voice-clone model (wav2vec2)",
        "tier": "primary", "score": round(p, 3), "confidence": 0.85, "weight": 2.5,
        "reasoning": explanation,
    }]
    extras = {
        "language": language, "speechChunks": result.speech_chunks,
        "totalChunks": result.total_chunks, "chunkResults": result.chunk_results,
    }
    verdict = "AI_GENERATED" if p > 0.5 else "HUMAN"
    return build_envelope(
        media_type="voice", verdict=verdict, confidence=confidence, ai_probability=p,
        explanation=explanation, signals=signals, extras=extras, model_backed=True,
        calibrated=False, escalate=(confidence == "Low"), filename=filename,
    )


if __name__ == "__main__":
    # ponytail: self-check — the risk/recommendation mapping must never advise paying on a fake.
    r, rec = risk_and_recommendation(V_AI, "High", False)
    assert r == RISK_HIGH and "Do NOT" in rec
    r, _ = risk_and_recommendation(V_HUMAN, "High", True)  # escalate bumps low->medium
    assert r == RISK_MEDIUM
    assert normalize_verdict("AI Generated") == V_AI
    assert normalize_verdict("Human Voice") == V_HUMAN
    assert normalize_verdict("whatever") == V_INCONCLUSIVE
    env = build_envelope(media_type="image", verdict="AI Generated", confidence="High",
                         ai_probability=0.9, explanation="x", model_backed=True, calibrated=True)
    assert env["verdict"] == V_AI and env["riskLevel"] == RISK_HIGH
    print("report envelope self-check OK")
