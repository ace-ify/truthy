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
)

from core.image.analyzers.base import AnalyzerResult


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


class Judge:
    """Aggregates analyzer results into a final verdict."""

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
        """Weighted average with majority consensus and conflict detection."""

        # Compute weighted score
        weighted_sum = 0.0
        weight_total = 0.0
        for r in valid_results:
            # Effective weight = analyzer weight * confidence
            w = r.weight * r.confidence
            weighted_sum += r.signal_score * w
            weight_total += w

        if weight_total > 0:
            avg_score = weighted_sum / weight_total
        else:
            avg_score = 0.5

        # --- Majority consensus detection ---
        # Count effective weight on each side
        ai_weight = 0.0  # analyzers scoring > 0.6 (leaning AI)
        human_weight = 0.0  # analyzers scoring < 0.4 (leaning Human)
        ai_scores = []
        human_scores = []
        for r in valid_results:
            ew = r.weight * r.confidence
            if r.signal_score > 0.6:
                ai_weight += ew
                ai_scores.append(r.signal_score)
            elif r.signal_score < 0.4:
                human_weight += ew
                human_scores.append(r.signal_score)

        # If one side has 2x+ the effective weight of the other, trust the majority
        has_ai_consensus = ai_weight > 0 and ai_weight > human_weight * 1.5
        has_human_consensus = human_weight > 0 and human_weight > ai_weight * 1.5

        if has_ai_consensus and ai_scores:
            # Majority says AI — compute score from AI-side only, blend with overall
            ai_avg = np.mean(ai_scores)
            # Pull toward AI consensus (60% consensus, 40% overall)
            consensus_score = ai_avg * 0.6 + avg_score * 0.4
            final_score = max(avg_score, consensus_score)
        elif has_human_consensus and human_scores:
            # Majority says Human
            human_avg = np.mean(human_scores)
            consensus_score = human_avg * 0.6 + avg_score * 0.4
            final_score = min(avg_score, consensus_score)
        else:
            final_score = avg_score

        # --- Override rules ---
        # Only let the CNN detector (high-weight, purpose-trained) override
        # Forensic heuristics and CLIP are too unreliable for overrides
        max_cnn_score = 0.0
        cnn_face_score = None
        cnn_method = None
        for r in valid_results:
            if (
                r.weight >= 2.0
                and r.confidence >= 0.7
                and r.signal_score > max_cnn_score
            ):
                max_cnn_score = r.signal_score
            # Track face model score and method separately
            if r.analyzer_name == "cnn_detector":
                cnn_face_score = r.details.get("face_score", None)
                cnn_method = r.details.get("method", None)

        # Strong AI signal from CNN
        if max_cnn_score > 0.85:
            final_score = max(final_score, max_cnn_score * 0.6 + final_score * 0.4)
        # Strong human signal from CNN — both models agree the image is real.
        # IMPORTANT: Only apply when cnn_method=="both_real" to avoid false floors
        # when CNN timed out, is uncertain, or models disagree. Without this guard
        # max_cnn_score==0 (from timeout/low-confidence) would trigger the floor
        # on AI images where heuristics alone aren't strong enough.
        elif cnn_method == "both_real" and max_cnn_score < 0.15 and avg_score < 0.4:
            final_score = min(final_score, 0.15)

        # --- CNN + ELA tiebreaker ---
        # Uses ELA to detect AI images that fool CNN models.
        # Face model (dima806) is highly reliable for portraits (face_score < 0.1 = portrait).
        # When a portrait has strongly elevated ELA, it's likely AI-generated.
        # IMPORTANT: Thresholds must be high enough to avoid false positives on
        # messaging-app-compressed images (WhatsApp/Telegram strip EXIF and
        # recompress at Q70-80, which naturally elevates ELA and destroys noise).
        cnn_result = None
        ela_result = None
        for r in valid_results:
            if r.analyzer_name == "cnn_detector":
                cnn_result = r
            if r.analyzer_name == "ela":
                ela_result = r

        tiebreaker_resolved = False

        if cnn_result and ela_result:
            face_score = cnn_result.details.get("face_score", 0.5)
            ai_model_score = cnn_result.details.get("ai_score", 0.5)
            ela_score = ela_result.signal_score
            raw_mean_ela = ela_result.details.get("mean_ela", 0.0)
            q_ratio = ela_result.details.get("q_ratio", 0.0)

            if (
                face_score < 0.1
                and ai_model_score > 0.8
                and ela_score >= 0.325
                and raw_mean_ela >= 0.8
            ):
                # CNN models disagree + ELA supports AI + high raw error level
                # raw_mean_ela >= 0.8 prevents false positives on messaging-compressed images
                final_score = 0.55 + ela_score * 0.3
                tiebreaker_resolved = True
            elif (
                face_score < 0.1
                and ela_score >= 0.34
                and raw_mean_ela >= 0.8
                and q_ratio > 0.75
            ):
                # Face model says real but ELA elevated + multi-quality confirms AI
                final_score = 0.55 + ela_score * 0.3
                tiebreaker_resolved = True
            elif face_score < 0.1 and ela_score < 0.20:
                # Face model says real + ELA very low → definitely real
                final_score = min(final_score, 0.25)
                tiebreaker_resolved = True
            elif (
                0.30 <= face_score <= 0.6
                and ai_model_score < 0.1
                and ela_score >= 0.325
                and raw_mean_ela >= 0.8
            ):
                # Face model uncertain + Ateeqq says real + ELA elevated + high raw error
                final_score = 0.55 + ela_score * 0.25
                tiebreaker_resolved = True
            elif (
                0.1 <= face_score <= 0.6
                and ai_model_score > 0.5
                and ela_score >= 0.35
                and raw_mean_ela >= 0.8
            ):
                # Face model uncertain + Ateeqq says AI + ELA supports AI + high raw error
                final_score = 0.55 + ela_score * 0.25
                tiebreaker_resolved = True

        # --- Conflict detection ---
        # Only flag conflict when there's NO clear majority consensus
        # and the tiebreaker hasn't already resolved the disagreement
        scores = [r.signal_score for r in valid_results if r.confidence >= 0.4]
        if len(scores) >= 2:
            score_std = np.std(scores)
            has_conflict = (
                score_std > 0.25
                and not has_ai_consensus
                and not has_human_consensus
                and not tiebreaker_resolved
            )
        else:
            has_conflict = False
            score_std = 0.0

        # --- Determine verdict ---
        if has_conflict and 0.3 < final_score < 0.7:
            verdict_str = "Inconclusive"
        elif final_score > IMAGE_AI_THRESHOLD:
            verdict_str = "AI Generated"
        else:
            verdict_str = "Human Created"

        # --- Confidence level ---
        distance = abs(final_score - IMAGE_AI_THRESHOLD)
        if has_conflict:
            confidence_str = "Low"
        elif distance > 0.3:
            confidence_str = "High"
        elif distance > 0.15:
            confidence_str = "Medium"
        else:
            confidence_str = "Low"

        # --- Explanation ---
        explanation = self._build_explanation(
            valid_results, final_score, verdict_str, has_conflict
        )

        # --- Extract heatmap from TruFor if available ---
        heatmap = self._extract_heatmap(valid_results)

        return ImageVerdict(
            overall_ai_probability=round(float(final_score), 3),
            verdict=verdict_str,
            confidence=confidence_str,
            explanation=explanation,
            signal_breakdown=self._build_breakdown(all_results),
            heatmap_b64=heatmap,
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
        results: List[AnalyzerResult],
        score: float,
        verdict: str,
        has_conflict: bool,
    ) -> str:
        """Build a human-readable explanation from analyzer results."""
        # Sort by how strongly they contributed
        strong_ai = [r for r in results if r.signal_score > 0.6 and r.confidence >= 0.4]
        strong_human = [
            r for r in results if r.signal_score < 0.4 and r.confidence >= 0.4
        ]

        parts = []
        if verdict == "AI Generated":
            ai_reasons = [r.reasoning for r in strong_ai[:3]]
            parts.append(
                f"Image shows strong indicators of AI generation ({score:.0%} probability)."
            )
            if ai_reasons:
                parts.append("Key signals: " + "; ".join(ai_reasons) + ".")
        elif verdict == "Human Created":
            human_reasons = [r.reasoning for r in strong_human[:3]]
            parts.append(
                f"Image appears to be human-created ({1 - score:.0%} probability)."
            )
            if human_reasons:
                parts.append("Supporting evidence: " + "; ".join(human_reasons) + ".")
        else:
            parts.append(
                "Analysis is inconclusive — analyzers show conflicting signals."
            )
            if strong_ai:
                parts.append(f"AI indicators: {strong_ai[0].reasoning}.")
            if strong_human:
                parts.append(f"Human indicators: {strong_human[0].reasoning}.")

        if has_conflict:
            parts.append(
                "Note: significant disagreement between analyzers reduces confidence."
            )

        return " ".join(parts)

    def _build_breakdown(self, results: List[AnalyzerResult]) -> List[dict]:
        """Convert analyzer results to serializable breakdown."""
        return [
            {
                "analyzer": r.analyzer_name,
                "display_name": r.display_name,
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
