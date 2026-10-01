"""
Video Deepfake Pipeline Orchestrator.

Combines:
1. Voice-clone detection on the audio track (wav2vec2) — the PRIMARY, model-backed signal.
2. Biological rPPG optical pulse (experimental supporting signal).
3. Audio-visual lip-sync heuristic (experimental supporting signal).

Only the voice-clone check is reliable enough to drive a verdict on its own; the visual
signals can lower confidence or trigger human review but never declare a deepfake alone.
"""

import os
import cv2
import logging
import tempfile
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from core.video.rppg import RPPGDetector
from core.video.phoneme_viseme import PhonemeVisemeAnalyzer
from core.audio_processor import process_uploaded_file

logger = logging.getLogger(__name__)


@dataclass
class VideoVerdict:
    overall_ai_probability: float
    verdict: str  # "AI_GENERATED" | "HUMAN" | "INCONCLUSIVE"
    confidence: str  # "High" | "Medium" | "Low"
    explanation: str
    fps: float
    duration_seconds: float
    total_frames: int
    rppg_results: Dict[str, Any]
    lip_sync_results: Dict[str, Any]
    voice_results: Dict[str, Any] = field(default_factory=dict)
    bvp_chart_b64: Optional[str] = None
    model_backed: bool = False
    escalate: bool = False


class VideoDetectionPipeline:
    """Orchestrates video forensics.

    Honesty model: the ONLY model-backed signal on a video is the wav2vec2 voice-clone
    check on the audio track. rPPG and lip-sync are weak, experimental *supporting*
    signals. So a confident AI verdict comes from the voice; the visual heuristics can
    lower confidence / trigger escalation but cannot, by themselves, declare a deepfake.
    """

    def __init__(self):
        self.rppg = RPPGDetector()
        self.lip_sync = PhonemeVisemeAnalyzer()

    def _analyze_voice(self, audio, sr: int) -> Dict[str, Any]:
        """Run the audio track through VAD + wav2vec2 voice-clone detector (primary signal)."""
        try:
            from core.vad import get_vad_processor
            from core.detector import get_detector
            from core.aggregator import aggregate_results
            from core.audio_processor import split_into_chunks
            from config import CHUNK_DURATION

            chunks = split_into_chunks(audio, sr, CHUNK_DURATION)
            vad = get_vad_processor()
            vad_results = vad.process_chunks(chunks, sr)
            det = get_detector()
            detection_results = []
            for i, (chunk, vr) in enumerate(zip(chunks, vad_results)):
                if vr["has_speech"]:
                    ai_prob, pred = det.predict(chunk, sr)
                    detection_results.append(
                        {"chunk_index": i, "ai_probability": ai_prob, "prediction": pred}
                    )
            if not detection_results:
                return {"evaluated": False, "reason": "No speech detected in the audio track."}
            ares = aggregate_results(vad_results, detection_results, CHUNK_DURATION)
            return {
                "evaluated": True,
                "voice_ai_prob": float(ares.overall_ai_probability),
                "speech_chunks": ares.speech_chunks,
                "reasoning": (
                    f"Voice-clone model flags the speech as likely AI-generated "
                    f"({ares.overall_ai_probability:.0%})."
                    if ares.overall_ai_probability > 0.5
                    else f"Voice-clone model finds the speech consistent with a real human voice "
                    f"({1 - ares.overall_ai_probability:.0%})."
                ),
            }
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Voice-clone analysis on video audio failed: {e}")
            return {"evaluated": False, "reason": f"Voice analysis unavailable: {e}"}

    def analyze_video(self, video_path: str, max_frames: int = 150) -> VideoVerdict:
        """
        Process a video file and evaluate biological and kinematic integrity.
        Args:
            video_path: Path to video file on disk
            max_frames: Max frames to process (balances speed with accuracy; 150 frames = 5s at 30fps)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 25.0

        total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frame_count / fps if total_frame_count > 0 else 0.0

        frames = []
        frame_idx = 0

        # Read frames sequentially up to max_frames
        while cap.isOpened() and len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            # Downsample frame resolution if very large to conserve CPU RAM
            h, w = frame.shape[:2]
            if max(h, w) > 720:
                scale = 720.0 / max(h, w)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            frames.append(frame)
            frame_idx += 1

        cap.release()

        if len(frames) < 15:
            raise ValueError("Video contains fewer than 15 valid frames.")

        # Extract embedded audio track using audio_processor (via librosa/pydub/ffmpeg)
        audio = None
        sr = 16000
        try:
            audio, sr = process_uploaded_file(video_path, remove_bg_noise=False)
        except Exception as e:
            logger.info(f"Video does not have readable audio track or audio extraction failed: {e}")

        # 1. Biological rPPG pulse analysis (SUPPORTING, experimental)
        rppg_res = self.rppg.analyze_frames(frames, fps)

        # 2. Audio-Visual lip-sync heuristic (SUPPORTING, experimental)
        lip_res = self.lip_sync.analyze_synchrony(frames, fps, audio, sr)

        # 3. Voice-clone detection on the audio track (PRIMARY, model-backed)
        voice_res = (
            self._analyze_voice(audio, sr)
            if audio is not None and len(audio) > 0
            else {"evaluated": False, "reason": "No audio track found in the video."}
        )

        # 4. Tiered fusion (mirrors the image judge's honesty rules)
        supporting_scores, supporting_weights = [], []
        if rppg_res["has_face"]:
            supporting_scores.append(rppg_res["rppg_ai_score"])
            supporting_weights.append(0.5)
        if lip_res["lip_sync_evaluated"]:
            supporting_scores.append(lip_res["kinematic_mismatch_score"])
            supporting_weights.append(0.5)
        support_mean = (
            float(np.average(supporting_scores, weights=supporting_weights))
            if supporting_scores else None
        )
        support_flags_synthetic = support_mean is not None and support_mean > 0.6

        model_backed = voice_res.get("evaluated", False)
        escalate = False

        if model_backed:
            # Primary voice signal drives the verdict; supporting nudges within ±0.15.
            final_prob = voice_res["voice_ai_prob"]
            if support_mean is not None:
                nudge = float(np.clip((support_mean - 0.5) * 0.5, -0.15, 0.15))
                final_prob = float(np.clip(final_prob + nudge, 0.0, 1.0))

            if final_prob > 0.55:
                verdict = "AI_GENERATED"
                explanation = (
                    f"Likely manipulated. {voice_res['reasoning']} "
                    "Do not act on payment instructions in this video without confirming on a trusted channel."
                )
            elif final_prob < 0.45:
                # Voice looks human — but our visual deepfake signals are weak. If they
                # raise a strong flag, we escalate instead of clearing the video.
                if support_flags_synthetic:
                    verdict = "INCONCLUSIVE"
                    escalate = True
                    explanation = (
                        f"The voice sounds like a real human ({1 - final_prob:.0%}), but visual signals "
                        "(pulse / lip-sync) look off. Face-video manipulation can't be reliably confirmed here — manual review recommended."
                    )
                else:
                    verdict = "HUMAN"
                    explanation = (
                        f"No strong signs of manipulation. {voice_res['reasoning']} "
                        "Note: visual deepfake checks here are limited; verify high-value requests independently."
                    )
            else:
                verdict = "INCONCLUSIVE"
                escalate = True
                explanation = "Borderline result — manual review recommended before acting."
        else:
            # No reliable primary signal. We will NOT declare a deepfake from weak heuristics.
            verdict = "INCONCLUSIVE"
            escalate = True
            final_prob = 0.5
            reason = voice_res.get("reason", "No audio to analyze.")
            explanation = (
                f"Could not run the reliable voice-clone check ({reason}). "
                "Only weak visual signals are available, which aren't enough to decide — manual review recommended. "
                f"{rppg_res['reasoning']} {lip_res['reasoning']}"
            )

        # Confidence (never claim High from supporting-only or borderline)
        dev = abs(final_prob - 0.5)
        if verdict == "INCONCLUSIVE":
            confidence = "Low"
        elif model_backed and dev > 0.25:
            confidence = "High"
        elif dev > 0.10:
            confidence = "Medium"
        else:
            confidence = "Low"

        return VideoVerdict(
            overall_ai_probability=round(final_prob, 3),
            verdict=verdict,
            confidence=confidence,
            explanation=explanation,
            fps=round(float(fps), 2),
            duration_seconds=round(float(duration), 2),
            total_frames=len(frames),
            rppg_results=rppg_res,
            lip_sync_results=lip_res,
            voice_results=voice_res,
            bvp_chart_b64=rppg_res.get("bvp_chart_b64"),
            model_backed=model_backed,
            escalate=escalate,
        )


# Singleton instance
_video_pipeline = None

def get_video_pipeline() -> VideoDetectionPipeline:
    global _video_pipeline
    if _video_pipeline is None:
        _video_pipeline = VideoDetectionPipeline()
    return _video_pipeline
