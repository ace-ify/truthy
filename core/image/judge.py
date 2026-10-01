"""
Judge: Aggregates all analyzer signals into a final verdict.
Supports weighted ensemble, LLM judge, and hybrid modes.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    IMAGE_AI_THRESHOLD,
    JUDGE_MODE,
    GEMINI_API_KEY,
    JUDGE_LLM_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
    JUDGE_LLM_PROVIDER,
    CALIBRATION_PATH,
)

from core.image.analyzers.base import AnalyzerResult
from core.image.calibration import Calibrator


@dataclass
class ImageVerdict:
    """Final output of the image detection pipeline."""

    overall_ai_probability: float
    verdict: str  # "AI Generated" | "Human Created" | "Inconclusive"
    confidence: str  # "High" | "Medium" | "Low"
    explanation: str
    generator_guess: Optional[str] = None
    signal_breakdown: List[dict] = field(default_factory=list)
    heatmap_b64: Optional[str] = None
    calibrated: bool = False       # was the probability produced by a fitted calibrator?
    model_backed: bool = False     # did a primary (model) detector drive this verdict?
    escalate: bool = False         # should this route to human review?
    provenance: Optional[dict] = None  # C2PA / EXIF / quant provenance panel


# How confident the primary (CNN) must be for us to trust its verdict on its own.
# Below this we do NOT guess — we return Inconclusive and recommend manual review.
PRIMARY_TRUST_CONFIDENCE = 0.7
# Above this we treat the primary as "High" confidence and skip the supporting nudge.
PRIMARY_HIGH_CONFIDENCE = 0.85
# Maximum absolute nudge supporting (heuristic) signals may apply to the primary prob.
MAX_SUPPORTING_NUDGE = 0.15


class Judge:
    """Aggregates analyzer results into a final verdict.

    Honesty model (see plan): a single model-backed *primary* detector (the dual-CNN)
    drives the verdict via a calibrated probability. The heuristic *supporting* signals
    may nudge confidence by at most ±MAX_SUPPORTING_NUDGE, and only when the primary
    isn't already High-confidence — they can never flip a confident model verdict. When
    the primary is uncertain or missing, we return Inconclusive and escalate, rather
    than letting uncalibrated heuristics fabricate a verdict.
    """

    def __init__(self):
        self.calibrator = Calibrator.load(CALIBRATION_PATH)

    def _has_llm_provider(self) -> bool:
        """Check if any LLM provider is configured."""
        return bool(GROQ_API_KEY or GEMINI_API_KEY)

    def evaluate(self, results: List[AnalyzerResult], mode: str = None) -> ImageVerdict:
        mode = mode or JUDGE_MODE

        # Filter out failed analyzers (confidence = 0)
        valid_results = [r for r in results if r.confidence > 0 and r.error is None]

        if not valid_results:
            return ImageVerdict(
                overall_ai_probability=0.5,
                verdict="Inconclusive",
                confidence="Low",
                explanation="No analyzers produced valid results.",
                signal_breakdown=self._build_breakdown(results),
            )

        if mode == "llm" and self._has_llm_provider():
            return self._llm_judge(valid_results, results)
        elif mode == "hybrid" and self._has_llm_provider():
            # Compute weighted score first; escalate to LLM if borderline
            verdict = self._weighted_ensemble(valid_results, results)
            if 0.35 <= verdict.overall_ai_probability <= 0.65:
                return self._llm_judge(valid_results, results)
            return verdict
        else:
            return self._weighted_ensemble(valid_results, results)

    def _weighted_ensemble(
        self, valid_results: List[AnalyzerResult], all_results: List[AnalyzerResult]
    ) -> ImageVerdict:
        """Tiered fusion: calibrated primary verdict + bounded supporting nudge."""
        threshold = IMAGE_AI_THRESHOLD

        primary = next(
            (r for r in valid_results if r.tier == "primary" and r.confidence > 0),
            None,
        )
        supporting = [
            r for r in valid_results
            if r.tier != "primary" and r.confidence >= 0.4 and r.weight > 0
        ]

        # No trustworthy model-backed signal -> do NOT guess from heuristics.
        if primary is None or primary.confidence < PRIMARY_TRUST_CONFIDENCE:
            return self._inconclusive(primary, supporting, all_results, valid_results)

        cal_prob = self.calibrator.apply(primary.signal_score)
        final = cal_prob

        # Supporting (heuristic) signals nudge confidence only when the primary is not
        # already High-confidence, and only within a hard bound. They cannot flip it.
        used_support = False
        if primary.confidence < PRIMARY_HIGH_CONFIDENCE and supporting:
            w = sum(r.weight * r.confidence for r in supporting)
            s_mean = (
                sum(r.signal_score * r.weight * r.confidence for r in supporting) / w
                if w else 0.5
            )
            nudge = float(np.clip((s_mean - 0.5) * 0.5, -MAX_SUPPORTING_NUDGE, MAX_SUPPORTING_NUDGE))
            final = float(np.clip(final + nudge, 0.0, 1.0))
            used_support = abs(nudge) > 0.01

        verdict_str = "AI Generated" if final > threshold else "Human Created"
        distance = abs(final - threshold)
        if not self.calibrator.calibrated:
            confidence_str = "Medium" if distance > 0.25 else "Low"  # never claim High uncalibrated
        elif distance > 0.3:
            confidence_str = "High"
        elif distance > 0.15:
            confidence_str = "Medium"
        else:
            confidence_str = "Low"

        explanation = self._build_explanation(primary, supporting, final, verdict_str, used_support)
        return ImageVerdict(
            overall_ai_probability=round(float(final), 3),
            verdict=verdict_str,
            confidence=confidence_str,
            explanation=explanation,
            signal_breakdown=self._build_breakdown(all_results),
            heatmap_b64=self._extract_heatmap(valid_results),
            calibrated=self.calibrator.calibrated,
            model_backed=True,
            escalate=(confidence_str == "Low"),
        )

    def _inconclusive(
        self,
        primary: Optional[AnalyzerResult],
        supporting: List[AnalyzerResult],
        all_results: List[AnalyzerResult],
        valid_results: List[AnalyzerResult],
    ) -> ImageVerdict:
        """Honest fallback when no confident model signal exists — flag for review."""
        if primary is not None:
            why = (
                f"The AI image classifier was not confident enough to decide "
                f"({primary.reasoning})."
            )
        else:
            why = "The AI image classifier did not return a usable result."
        explanation = (
            f"Inconclusive — {why} Supporting forensic signals are not reliable enough "
            "to decide on their own, so this needs manual review before you act on it."
        )
        return ImageVerdict(
            overall_ai_probability=0.5,
            verdict="Inconclusive",
            confidence="Low",
            explanation=explanation,
            signal_breakdown=self._build_breakdown(all_results),
            heatmap_b64=self._extract_heatmap(valid_results),
            calibrated=self.calibrator.calibrated,
            model_backed=False,
            escalate=True,
        )


    def _llm_judge(
        self, valid_results: List[AnalyzerResult], all_results: List[AnalyzerResult]
    ) -> ImageVerdict:
        """Use an LLM to reason about analyzer results."""
        # Build structured prompt
        signal_summary = "\n".join(
            f"- {r.display_name}: score={r.signal_score:.2f}, "
            f"confidence={r.confidence:.2f}, reasoning: {r.reasoning}"
            for r in valid_results
        )

        prompt = f"""You are an expert forensic image analyst. Multiple detection systems have analyzed an image.
Your job: synthesize all signals into a final verdict.

## Analyzer Results:
{signal_summary}

## Instructions:
1. Weigh each analyzer by its confidence level
2. If analyzers conflict, reason about which signals are more reliable
3. Consider that frequency analysis and noise analysis are more fundamental than others
4. CNN classifier results should be weighted heavily but can be wrong on out-of-distribution images

Respond in EXACTLY this JSON format (no other text):
{{
  "ai_probability": 0.XX,
  "verdict": "AI Generated" or "Human Created" or "Inconclusive",
  "confidence": "High" or "Medium" or "Low",
  "explanation": "2-3 sentence explanation of your reasoning",
  "generator_guess": "Likely DALL-E/Midjourney/Stable Diffusion/etc" or null
}}"""

        # Pick provider
        provider = JUDGE_LLM_PROVIDER
        if provider == "auto":
            provider = "groq" if GROQ_API_KEY else "gemini"

        try:
            if provider == "groq":
                text = self._call_groq(prompt)
            else:
                text = self._call_gemini(prompt)

            # Parse JSON from response
            import json

            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            parsed = json.loads(text)

            return ImageVerdict(
                overall_ai_probability=round(float(parsed["ai_probability"]), 3),
                verdict=parsed["verdict"],
                confidence=parsed["confidence"],
                explanation=parsed["explanation"],
                generator_guess=parsed.get("generator_guess"),
                signal_breakdown=self._build_breakdown(all_results),
            )

        except Exception as e:
            # Fallback to weighted ensemble if LLM fails
            verdict = self._weighted_ensemble(valid_results, all_results)
            verdict.explanation += f" (LLM judge unavailable: {e})"
            return verdict

    def _call_groq(self, prompt: str) -> str:
        """Call Groq API."""
        from groq import Groq

        api_key = GROQ_API_KEY.strip()
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()

    def _call_gemini(self, prompt: str) -> str:
        """Call Gemini API."""
        from google import genai

        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model=JUDGE_LLM_MODEL,
            contents=prompt,
        )
        return response.text.strip()

    def _build_explanation(
        self,
        primary: AnalyzerResult,
        supporting: List[AnalyzerResult],
        score: float,
        verdict: str,
        used_support: bool,
    ) -> str:
        """Plain, honest explanation grounded in the primary model + optional support."""
        cal_note = "" if self.calibrator.calibrated else " (confidence not yet calibrated)"
        if verdict == "AI Generated":
            head = f"Likely AI-generated ({score:.0%} probability){cal_note}."
        else:
            head = f"Likely authentic / human-created ({1 - score:.0%} probability){cal_note}."
        parts = [head, f"Primary detector: {primary.reasoning}"]
        if used_support and supporting:
            top = sorted(supporting, key=lambda r: abs(r.signal_score - 0.5), reverse=True)[:2]
            parts.append(
                "Supporting forensic signals ("
                + "; ".join(f"{r.display_name} {r.signal_score:.0%}" for r in top)
                + ") slightly adjusted the confidence."
            )
        return " ".join(parts)

    def _build_breakdown(self, results: List[AnalyzerResult]) -> List[dict]:
        """Convert analyzer results to serializable breakdown."""
        return [
            {
                "analyzer": r.analyzer_name,
                "display_name": r.display_name,
                "tier": getattr(r, "tier", "supporting"),
                "score": round(r.signal_score, 3),
                "confidence": round(r.confidence, 3),
                "weight": round(r.weight, 2),
                "reasoning": r.reasoning,
                "processing_time_ms": r.processing_time_ms,
                "error": r.error,
            }
            for r in results
        ]

    def _extract_heatmap(self, results: List[AnalyzerResult]) -> Optional[str]:
        """Extract heatmap_b64 from TruFor result if present."""
        for r in results:
            if r.analyzer_name == "trufor_heatmap" and r.details:
                heatmap = r.details.get("heatmap_b64")
                if heatmap:
                    return heatmap
        return None


# Singleton
_judge = None


def get_judge() -> Judge:
    global _judge
    if _judge is None:
        _judge = Judge()
    return _judge
