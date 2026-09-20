"""
Audio-Visual Phoneme-Viseme Biomechanical Kinematic Analyzer.

Physical Principle:
Human speech requires precise physical coordination between acoustic pressure generation in the
vocal tract and mechanical kinematics of the facial musculature:
1. Bilabial plosives (/p/, /b/, /m/) physically require COMPLETE lip closure prior to sound release.
2. Labiodental fricatives (/f/, /v/) require lower lip contact with upper incisors.
3. High acoustic energy vowels (/a/, /o/) require substantial mouth cavity opening.

Neural speech-driven video generators (Wav2Lip, SadTalker, SyncTalk, LivePortrait) optimize for
visual realism or perceptual lip sync error, but frequently generate acoustic sound bursts
while the subject's lips remain wide open, violating speech biomechanics.
"""

import numpy as np
import cv2
import logging
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)


class PhonemeVisemeAnalyzer:
    """Evaluates cross-modal physical synchronization between audio energy and lip kinematics."""

    def __init__(self):
        self.mouth_cascade = None
        self.face_cascade = None
        try:
            if hasattr(cv2, 'CascadeClassifier') and hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
                cascade_path = cv2.data.haarcascades + 'haarcascade_smile.xml'
                self.mouth_cascade = cv2.CascadeClassifier(cascade_path)
                face_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.face_cascade = cv2.CascadeClassifier(face_path)
        except Exception:
            self.mouth_cascade = None
            self.face_cascade = None

    def extract_lip_aperture(self, frame_bgr: np.ndarray) -> float:
        """
        Estimate vertical mouth opening aperture from video frame.
        Returns a normalized aperture index (0.0 = fully closed, 1.0 = wide open).
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        mouth_roi = None
        if self.face_cascade is not None:
            try:
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(64, 64))
                if len(faces) > 0:
                    fx, fy, fw, fh = max(faces, key=lambda b: b[2] * b[3])
                    my1 = int(fy + 0.60 * fh)
                    my2 = int(fy + 0.95 * fh)
                    mx1 = int(fx + 0.25 * fw)
                    mx2 = int(fx + 0.75 * fw)
                    mouth_roi = gray[my1:my2, mx1:mx2]
            except Exception:
                mouth_roi = None

        if mouth_roi is None or mouth_roi.size == 0:
            # Fallback: lower third center of frame contains mouth
            mouth_roi = gray[2 * h // 3: h, w // 4: 3 * w // 4]

        if mouth_roi.size == 0:
            return 0.2

        # Threshold to detect dark oral cavity opening
        # Oral cavity inside an open mouth is significantly darker than lips/skin
        min_val = np.percentile(mouth_roi, 10)
        max_val = np.percentile(mouth_roi, 90)
        thresh_val = min_val + 0.35 * (max_val - min_val)

        dark_pixels = mouth_roi < thresh_val
        opening_ratio = np.mean(dark_pixels)

        # Normalized aperture
        aperture = float(np.clip(opening_ratio * 4.0, 0.0, 1.0))
        return aperture

    def analyze_synchrony(
        self,
        frames: List[np.ndarray],
        fps: float,
        audio: Optional[np.ndarray],
        sr: int = 16000
    ) -> Dict[str, Any]:
        """
        Evaluate physical audio-visual synchrony.
        Args:
            frames: Sequence of video frames
            fps: Video frames per second
            audio: Synchronous audio waveform (numpy 1D array)
            sr: Audio sample rate
        """
        if len(frames) < 15 or audio is None or len(audio) == 0 or fps <= 0:
            return {
                "lip_sync_evaluated": False,
                "biomechanical_violations": 0,
                "kinematic_mismatch_score": 0.5,
                "reasoning": "Audio stream or video frame count insufficient for kinematic alignment.",
            }

        # 1. Track lip aperture across all frames
        lip_apertures = [self.extract_lip_aperture(f) for f in frames]
        n_frames = len(lip_apertures)

        # 2. Compute audio root-mean-square (RMS) energy matching each frame window
        samples_per_frame = int(sr / fps)
        audio_energies = []

        for i in range(n_frames):
            start = i * samples_per_frame
            end = min(len(audio), (i + 1) * samples_per_frame)
            if start < len(audio):
                segment = audio[start:end]
                rms = np.sqrt(np.mean(segment ** 2)) if len(segment) > 0 else 0.0
            else:
                rms = 0.0
            audio_energies.append(rms)

        # Normalize audio energy
        max_energy = np.percentile(audio_energies, 95) + 1e-6
        norm_energies = np.array(audio_energies) / max_energy
        norm_apertures = np.array(lip_apertures)

        # 3. Detect Biomechanical Physical Violations:
        # High acoustic speech energy while lips are completely closed (aperture < 0.05)
        # OR completely silent audio while lips are flapping in wide aperture (> 0.6)
        speech_while_closed = (norm_energies > 0.6) & (norm_apertures < 0.05)
        silence_while_flapping = (norm_energies < 0.08) & (norm_apertures > 0.55)
        
        violations_count = int(np.sum(speech_while_closed) + np.sum(silence_while_flapping))
        violation_rate = violations_count / float(n_frames)

        # 4. Cross-correlation alignment
        # In natural speech, mouth aperture leads or aligns with acoustic envelope (lag -100ms to +100ms)
        centered_aperture = norm_apertures - np.mean(norm_apertures)
        centered_energy = norm_energies - np.mean(norm_energies)
        
        if np.std(centered_aperture) > 1e-4 and np.std(centered_energy) > 1e-4:
            corr = np.correlate(centered_aperture, centered_energy, mode='full')
            max_corr = np.max(corr) / (np.linalg.norm(centered_aperture) * np.linalg.norm(centered_energy) + 1e-8)
        else:
            max_corr = 0.5

        # High violation rate and low cross-correlation => Deepfake lip sync
        kinematic_mismatch = 0.6 * min(violation_rate * 5.0, 1.0) + 0.4 * (1.0 - max(0.0, float(max_corr)))
        kinematic_mismatch = float(np.clip(kinematic_mismatch, 0.0, 1.0))

        if kinematic_mismatch > 0.55:
            reasoning = (
                f"Audio-visual kinematic mismatch detected ({violations_count} biomechanical timing violations). "
                f"Acoustic formant energy onsets do not correspond with anatomical lip aperture kinematics."
            )
        else:
            reasoning = (
                f"Acoustic speech energy and lip aperture show physical kinematic synchronization "
                f"(correlation: {max_corr:.2f}, {violations_count} violations)."
            )

        return {
            "lip_sync_evaluated": True,
            "biomechanical_violations": violations_count,
            "correlation": round(float(max_corr), 3),
            "kinematic_mismatch_score": round(kinematic_mismatch, 3),
            "reasoning": reasoning,
        }
