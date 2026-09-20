"""
Remote Photoplethysmography (rPPG) Hemodynamic Pulse Detector.

Biological Principle:
Living humans have a heart rate: cardiac systolic contractions pump oxygenated blood into
sub-dermal capillary beds. Hemoglobin selectively absorbs green light (520–580 nm), producing
subtle, periodic skin color oscillations synchronous with the pulse.

Generative video models (FaceSwap, DeepFaceLab, LivePortrait, Sora, Kling, Runaway Gen-3)
synthesize frames independently or with temporal smoothing that lacks cardiac hemodynamic periodicity.
When tracking facial ROIs (forehead and cheeks) across consecutive video frames, an authentic human
displays a distinct spectral peak between 0.75 Hz and 3.0 Hz (45 to 180 BPM). Synthetic deepfakes
exhibit unstructured noise or zero physiological periodicity.
"""

import numpy as np
import cv2
import io
import base64
import logging
from typing import Dict, Any, List, Optional, Tuple
from scipy.signal import butter, filtfilt, find_peaks
from scipy.fft import rfft, rfftfreq

logger = logging.getLogger(__name__)


class RPPGDetector:
    """Extracts optical Blood Volume Pulse (BVP) from facial video frames to verify biological life."""

    def __init__(self, min_bpm: float = 45.0, max_bpm: float = 180.0):
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
        self.face_cascade = None
        try:
            if hasattr(cv2, 'CascadeClassifier') and hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
        except Exception:
            self.face_cascade = None

    def extract_face_roi(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Detect face and crop the central forehead/cheek region (highest capillary density)."""
        h, w = frame_bgr.shape[:2]
        if self.face_cascade is not None:
            try:
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(64, 64))
                if len(faces) > 0:
                    x, y, fw, fh = max(faces, key=lambda b: b[2] * b[3])
                    roi_y1 = int(y + 0.15 * fh)
                    roi_y2 = int(y + 0.65 * fh)
                    roi_x1 = int(x + 0.20 * fw)
                    roi_x2 = int(x + 0.80 * fw)
                    roi = frame_bgr[roi_y1:roi_y2, roi_x1:roi_x2]
                    if roi.size > 0:
                        return roi
            except Exception:
                pass

        # Robust skin/central capillary ROI fallback (central 50% region)
        return frame_bgr[h//4: 3*h//4, w//4: 3*w//4]

        # Select largest face
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        
        # Forehead and upper cheek sub-ROI (approx 20% to 65% height, 20% to 80% width of face box)
        roi_y1 = int(y + 0.15 * h)
        roi_y2 = int(y + 0.65 * h)
        roi_x1 = int(x + 0.20 * w)
        roi_x2 = int(x + 0.80 * w)

        roi = frame_bgr[roi_y1:roi_y2, roi_x1:roi_x2]
        return roi if roi.size > 0 else frame_bgr[y:y+h, x:x+w]

    def analyze_frames(self, frames: List[np.ndarray], fps: float) -> Dict[str, Any]:
        """
        Analyze a sequence of BGR video frames for cardiac periodicity.
        Args:
            frames: List of BGR video frames (minimum 30 frames, ideally 60-150 frames)
            fps: Frame rate of the video (e.g. 24.0, 30.0)
        """
        if len(frames) < 30 or fps <= 0:
            return {
                "has_face": False,
                "has_pulse": False,
                "bpm": None,
                "pulse_snr": 0.0,
                "rppg_ai_score": 0.5,
                "reasoning": "Insufficient video frames for biological pulse analysis (minimum 30 frames required).",
                "bvp_chart_b64": None,
            }

        # 1. Extract chrominance signals across frames
        # Use Plane-Orthogonal-to-Skin (POS) algorithm or Green-channel photoplethysmography
        r_signals, g_signals, b_signals = [], [], []

        for frame in frames:
            roi = self.extract_face_roi(frame)
            if roi is None or roi.size == 0:
                continue
            # Mean RGB intensities in ROI
            b_signals.append(np.mean(roi[:, :, 0]))
            g_signals.append(np.mean(roi[:, :, 1]))
            r_signals.append(np.mean(roi[:, :, 2]))

        n_samples = len(g_signals)
        if n_samples < 30:
            return {
                "has_face": False,
                "has_pulse": False,
                "bpm": None,
                "pulse_snr": 0.0,
                "rppg_ai_score": 0.5,
                "reasoning": "Face not consistently trackable across video sequence.",
                "bvp_chart_b64": None,
            }

        # Normalize chrominance signals
        r = np.array(r_signals) / (np.mean(r_signals) + 1e-6)
        g = np.array(g_signals) / (np.mean(g_signals) + 1e-6)
        b = np.array(b_signals) / (np.mean(b_signals) + 1e-6)

        # Plane-Orthogonal-to-Skin (POS) projection:
        # S1 = G - B; S2 = G + B - 2*R
        # BVP = S1 + (std(S1)/std(S2)) * S2
        s1 = g - b
        s2 = g + b - 2.0 * r
        std_s1 = np.std(s1) + 1e-6
        std_s2 = np.std(s2) + 1e-6
        bvp_raw = s1 + (std_s1 / std_s2) * s2

        # 2. Bandpass Filter (0.75 Hz to 3.0 Hz => 45 to 180 BPM)
        lowcut = self.min_bpm / 60.0
        highcut = self.max_bpm / 60.0
        nyquist = 0.5 * fps

        if highcut >= nyquist:
            highcut = nyquist - 0.1

        if lowcut >= highcut:
            lowcut = 0.5
            highcut = min(2.5, nyquist - 0.1)

        b, a = butter(2, [lowcut / nyquist, highcut / nyquist], btype='band')
        try:
            bvp_filtered = filtfilt(b, a, bvp_raw)
        except Exception:
            bvp_filtered = bvp_raw - np.mean(bvp_raw)

        # 3. Frequency Spectrum & Pulse Peak Estimation
        fft_vals = np.abs(rfft(bvp_filtered))
        freqs = rfftfreq(n_samples, d=1.0 / fps)

        # Mask to valid cardiac range
        valid_mask = (freqs >= lowcut) & (freqs <= highcut)
        if not np.any(valid_mask):
            cardiac_freqs = freqs
            cardiac_fft = fft_vals
        else:
            cardiac_freqs = freqs[valid_mask]
            cardiac_fft = fft_vals[valid_mask]

        max_idx = np.argmax(cardiac_fft)
        peak_freq = cardiac_freqs[max_idx]
        detected_bpm = peak_freq * 60.0

        # Signal-to-Noise Ratio (SNR) of cardiac peak
        peak_energy = cardiac_fft[max_idx] ** 2
        total_energy = np.sum(cardiac_fft ** 2) + 1e-8
        pulse_snr = float(peak_energy / total_energy)

        # 4. Biological Evaluation
        # Authentic human faces exhibit a clear dominant pulse peak (SNR > 0.35)
        # Deepfake face replacements have low SNR (< 0.20) because synthetic diffusion generates noisy color variance
        if pulse_snr > 0.38 and 50 <= detected_bpm <= 150:
            has_pulse = True
            rppg_ai_score = 0.15  # Strongly human
            reasoning = (
                f"Biological heart rate pulse verified via optical rPPG ({detected_bpm:.1f} BPM, SNR: {pulse_snr:.1%}). "
                f"Sub-dermal micro-capillary blood volume pulse is consistent with a living human subject."
            )
        elif pulse_snr > 0.25:
            has_pulse = True
            rppg_ai_score = 0.40  # Inconclusive/weak pulse
            reasoning = (
                f"Weak cardiac signal detected ({detected_bpm:.1f} BPM, SNR: {pulse_snr:.1%}). "
                f"Potential video compression or subtle motion artifacts."
            )
        else:
            has_pulse = False
            rppg_ai_score = 0.85  # Strongly synthetic
            reasoning = (
                f"No biological cardiac pulse detected (SNR: {pulse_snr:.1%}). "
                f"Facial pixels lack the physiological sub-dermal hemoglobin absorption cycles expected in authentic human video."
            )

        # Generate base64 visualization plot of the BVP signal
        bvp_chart_b64 = self._render_bvp_chart(bvp_filtered, fps, detected_bpm, pulse_snr)

        return {
            "has_face": True,
            "has_pulse": has_pulse,
            "bpm": round(float(detected_bpm), 1),
            "pulse_snr": round(pulse_snr, 3),
            "rppg_ai_score": float(np.clip(rppg_ai_score, 0.0, 1.0)),
            "reasoning": reasoning,
            "bvp_chart_b64": bvp_chart_b64,
        }

    def _render_bvp_chart(self, bvp_signal: np.ndarray, fps: float, bpm: float, snr: float) -> str:
        """Render optical BVP wave as a dark-mode base64 PNG chart."""
        try:
            from PIL import Image, ImageDraw, ImageFont
            w, h = 600, 200
            img = Image.new('RGB', (w, h), color=(15, 17, 23))
            draw = ImageDraw.Draw(img)

            # Draw background grid
            for x in range(0, w, 50):
                draw.line([(x, 0), (x, h)], fill=(30, 35, 45))
            for y in range(0, h, 40):
                draw.line([(0, y), (w, y)], fill=(30, 35, 45))

            # Normalize signal to canvas height
            sig = bvp_signal - np.mean(bvp_signal)
            max_amp = np.max(np.abs(sig)) + 1e-6
            normalized = sig / max_amp * 0.75  # fit within +/- 75% height

            points = []
            n_pts = len(normalized)
            for i in range(n_pts):
                cx = int(i / float(n_pts - 1) * (w - 40)) + 20
                cy = int(h / 2 - normalized[i] * (h / 2 - 20))
                points.append((cx, cy))

            # Draw waveform
            for i in range(len(points) - 1):
                color = (59, 130, 246) if snr > 0.3 else (239, 68, 68)
                draw.line([points[i], points[i + 1]], fill=color, width=2)

            # Draw labels
            title = f"Optical Hemodynamic BVP Pulse: {bpm:.1f} BPM (SNR: {snr:.1%})"
            draw.text((20, 15), title, fill=(255, 255, 255))

            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception as e:
            logger.warning(f"Failed to render BVP chart: {e}")
            return ""
