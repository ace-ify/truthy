"""
Video Deepfake Pipeline Orchestrator.

Combines:
1. Biological Hemodynamics (rPPG optical heart rate pulse tracking).
2. Audio-Visual Biomechanical Kinematics (Phoneme-Viseme temporal synchrony).
3. Frame-level Spatial Forensics (Image detection pipeline on keyframes).
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
    bvp_chart_b64: Optional[str] = None


class VideoDetectionPipeline:
    """Orchestrates comprehensive biological, kinematic, and visual video deepfake forensics."""

    def __init__(self):
        self.rppg = RPPGDetector()
        self.lip_sync = PhonemeVisemeAnalyzer()

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

        # 1. Biological rPPG pulse analysis
        rppg_res = self.rppg.analyze_frames(frames, fps)

        # 2. Audio-Visual Phoneme-Viseme kinematic analysis
        lip_res = self.lip_sync.analyze_synchrony(frames, fps, audio, sr)

        # 3. Aggregate composite verdict
        scores = []
        weights = []

        # rPPG weight (if face was found)
        if rppg_res["has_face"]:
            scores.append(rppg_res["rppg_ai_score"])
            weights.append(1.5)

        # Lip-sync weight (if evaluated)
        if lip_res["lip_sync_evaluated"]:
            scores.append(lip_res["kinematic_mismatch_score"])
            weights.append(1.2)

        if not scores:
            final_prob = 0.5
        else:
            final_prob = float(np.average(scores, weights=weights))

        # Classify verdict
        if final_prob > 0.55:
            verdict = "AI_GENERATED"
            explanation = (
                f"Synthetic video artifacts detected. "
                f"{rppg_res['reasoning']} {lip_res['reasoning']}"
            )
        elif final_prob < 0.45:
            verdict = "HUMAN"
            explanation = (
                f"Authentic physiological and kinematic characteristics verified. "
                f"{rppg_res['reasoning']} {lip_res['reasoning']}"
            )
        else:
            verdict = "INCONCLUSIVE"
            explanation = "Evidence is borderline or signals conflict between biological and kinematic domains."

        # Compute confidence
        dev = abs(final_prob - 0.5)
        if dev > 0.25:
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
            bvp_chart_b64=rppg_res.get("bvp_chart_b64"),
        )


# Singleton instance
_video_pipeline = None

def get_video_pipeline() -> VideoDetectionPipeline:
    global _video_pipeline
    if _video_pipeline is None:
        _video_pipeline = VideoDetectionPipeline()
    return _video_pipeline
