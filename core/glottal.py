"""
Glottal Flow Inverse Filtering & Acoustic Phase Jitter Forensics.

Physical Principle:
The human vocal apparatus creates sound by passing pulmonary airflow through vibrating vocal cords
(the glottis). Under the Liljencrants-Fant (LF) voice source model, the glottal pulse has an
asymmetric triangular shape: a smooth opening phase followed by an abrupt, discontinuous closing
instant (Glottal Closure Instant - GCI). The vocal tract (pharynx, oral, and nasal cavities) then
acts as a linear acoustic filter imposing resonant formant frequencies.

Neural Vocoders (HiFi-GAN, BigVGAN, MelGAN, WaveNet) in AI voice synthesis generators do NOT
model pulmonary biomechanics or glottal aerodynamics. Instead, they synthesize audio from
Mel-spectrograms by estimating STFT phases using convolutional upsamplers. This inevitably introduces:
1. Symmetric or chaotic glottal derivative pulses under LPC inverse filtering.
2. Unnatural Glottal Closure Instant (GCI) jitter and phase smearing around formant boundaries.
"""

import numpy as np
import logging
from typing import Dict, Tuple, List, Optional
from scipy.signal import lfilter, find_peaks

logger = logging.getLogger(__name__)


def solve_lpc(signal: np.ndarray, order: int = 16) -> np.ndarray:
    """
    Compute Linear Predictive Coding (LPC) coefficients via the Levinson-Durbin recursion.
    Auto-regressive model: s[n] = sum_{k=1}^p a_k s[n-k] + e[n]
    """
    if len(signal) < order + 1:
        return np.ones(order + 1)

    # Compute autocorrelation coefficients
    r = np.correlate(signal, signal, mode='full')
    mid = len(r) // 2
    r = r[mid:mid + order + 1]

    if r[0] <= 1e-12:
        return np.ones(order + 1)

    # Levinson-Durbin recursion
    a = np.zeros(order + 1)
    e = r[0]
    a[0] = 1.0

    for i in range(1, order + 1):
        if e <= 1e-12:
            break
        gamma = -np.dot(a[:i], r[i:0:-1]) / e
        a[1:i+1] += gamma * a[i-1::-1]
        a[i] = gamma
        e *= (1.0 - gamma ** 2)

    return a


class GlottalForensicAnalyzer:
    """
    Extracts glottal flow excitation through inverse vocal tract filtering and evaluates
    physiological speech markers.
    """

    def __init__(self, sample_rate: int = 16000, lpc_order: int = 16):
        self.sample_rate = sample_rate
        self.lpc_order = lpc_order

    def analyze_chunk(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Analyze an audio chunk (16kHz mono numpy array) for glottal flow biomechanical anomalies.
        Returns:
            Dict containing:
                - 'gci_jitter': Glottal Closure Instant timing irregularity
                - 'glottal_asymmetry_error': Deviation from human vocal fold opening/closing ratio
                - 'glottal_ai_probability': Estimated synthetic voice probability from acoustics
        """
        if len(audio) < self.sample_rate * 0.2:
            return {
                "gci_jitter": 0.0,
                "glottal_asymmetry_error": 0.0,
                "glottal_ai_probability": 0.5,
                "is_voiced": False,
            }

        # Normalize audio amplitude
        audio = audio - np.mean(audio)
        max_val = np.max(np.abs(audio))
        if max_val > 1e-6:
            audio = audio / max_val
        else:
            return {
                "gci_jitter": 0.0,
                "glottal_asymmetry_error": 0.0,
                "glottal_ai_probability": 0.5,
                "is_voiced": False,
            }

        # 1. Estimate Vocal Tract Filter via LPC
        # Frame-by-frame analysis (30ms window, 15ms hop)
        frame_len = int(0.030 * self.sample_rate)
        hop_len = int(0.015 * self.sample_rate)
        num_frames = (len(audio) - frame_len) // hop_len

        if num_frames < 3:
            return {
                "gci_jitter": 0.0,
                "glottal_asymmetry_error": 0.0,
                "glottal_ai_probability": 0.5,
                "is_voiced": False,
            }

        gci_periods: List[float] = []
        asymmetry_ratios: List[float] = []

        for f_idx in range(num_frames):
            start = f_idx * hop_len
            frame = audio[start:start + frame_len] * np.hamming(frame_len)

            # Voicing energy check
            if np.std(frame) < 0.02:
                continue

            # Inverse filtering: e[n] = A(z) * S(z)
            lpc_coeffs = solve_lpc(frame, order=self.lpc_order)
            glottal_residual = lfilter(lpc_coeffs, [1.0], frame)

            # Derivative of glottal flow typically shows sharp negative peaks at GCI
            peaks, props = find_peaks(-glottal_residual, distance=int(self.sample_rate / 400), prominence=0.1)
            
            if len(peaks) >= 2:
                periods = np.diff(peaks) / float(self.sample_rate)
                # Keep periods within realistic human pitch (50 Hz - 350 Hz => 2.8ms - 20ms)
                valid_periods = [p for p in periods if 0.0028 <= p <= 0.020]
                gci_periods.extend(valid_periods)

                # Measure asymmetry of pulses
                for p_idx in range(len(peaks) - 1):
                    p_start = peaks[p_idx]
                    p_end = peaks[p_idx + 1]
                    pulse = glottal_residual[p_start:p_end]
                    if len(pulse) > 6:
                        min_idx = np.argmin(pulse)
                        # Opening duration vs closing duration
                        opening = max(1, min_idx)
                        closing = max(1, len(pulse) - min_idx)
                        ratio = opening / float(closing)
                        asymmetry_ratios.append(ratio)

        if len(gci_periods) < 5 or not asymmetry_ratios:
            return {
                "gci_jitter": 0.0,
                "glottal_asymmetry_error": 0.0,
                "glottal_ai_probability": 0.5,
                "is_voiced": False,
            }

        # 2. Compute Glottal Metrics
        # Jitter: Local pitch period perturbation: mean(|T_i - T_{i-1}|) / mean(T)
        diffs = np.abs(np.diff(gci_periods))
        mean_period = np.mean(gci_periods)
        jitter = float(np.mean(diffs) / (mean_period + 1e-8))

        # Natural human speech has subtle natural jitter (0.005 - 0.03)
        # Deepfake vocoders either show hyper-robotic zero jitter (< 0.004) OR erratic chaotic jitter (> 0.08)
        jitter_anomaly = 0.0
        if jitter < 0.004:
            jitter_anomaly = min((0.004 - jitter) / 0.004 * 0.8 + 0.2, 1.0)  # Overly synthetic/robotic
        elif jitter > 0.06:
            jitter_anomaly = min((jitter - 0.06) / 0.06 * 0.7 + 0.3, 1.0)  # Vocoder phase artifacts
        else:
            jitter_anomaly = 0.15  # Natural human range

        # Asymmetry deviation: Natural human LF model has an asymmetry ratio around 1.5 - 3.5
        mean_asymmetry = float(np.median(asymmetry_ratios))
        asymmetry_err = 0.0
        if mean_asymmetry < 1.1 or mean_asymmetry > 4.5:
            asymmetry_err = 0.7
        else:
            asymmetry_err = 0.2

        ai_probability = 0.6 * jitter_anomaly + 0.4 * asymmetry_err
        ai_probability = float(np.clip(ai_probability, 0.0, 1.0))

        return {
            "gci_jitter": float(jitter),
            "glottal_asymmetry_ratio": float(mean_asymmetry),
            "glottal_ai_probability": ai_probability,
            "is_voiced": True,
        }
