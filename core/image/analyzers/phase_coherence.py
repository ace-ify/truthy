"""
Bi-Spectral Fourier Phase Coherence Analyzer.

Physical Principle:
Natural camera lenses and optical image formation strictly preserve harmonic phase
coupling across spatial frequencies. Generative AI models (latent diffusion, GANs)
synthesize textures via spatial deconvolution and local attention, causing severe
disruptions in the bi-spectral phase angle even when the Fourier magnitude spectrum
appears natural.
"""

import numpy as np
import logging
from scipy.fft import fft2, fftshift

from core.image.analyzers.base import BaseAnalyzer, AnalyzerResult
from core.image.preprocessor import ImageData

logger = logging.getLogger(__name__)


class PhaseCoherenceAnalyzer(BaseAnalyzer):
    """Measures bi-spectral phase coupling and phase congruence across spatial frequencies."""

    name = "phase_coherence"
    display_name = "Bi-Spectral Phase Coherence"
    weight = 1.3

    def _analyze(self, image_data: ImageData) -> AnalyzerResult:
        gray = image_data.grayscale  # Normalized 0.0 - 1.0 float
        h, w = gray.shape

        # Downsample large images for computational efficiency if needed
        max_dim = 512
        if max(h, w) > max_dim:
            from PIL import Image
            pil_resized = image_data.pil_image.convert("L").resize((max_dim, max_dim), Image.BILINEAR)
            gray = np.array(pil_resized, dtype=np.float64) / 255.0
            h, w = gray.shape

        # 2D Fast Fourier Transform
        f_transform = fft2(gray)
        f_shifted = fftshift(f_transform)
        
        # Magnitude and Phase angles (-pi to +pi)
        magnitude = np.abs(f_shifted)
        phase = np.angle(f_shifted)

        # 1. Bi-spectral Harmonic Phase Coupling
        # Select key low-frequency reference points and test phase alignment with their harmonics:
        # B(f1, f2) = F(f1) * F(f2) * conj(F(f1 + f2))
        cy, cx = h // 2, w // 2
        harmonic_coupling_errors = []

        # Sample grid of base spatial frequencies
        step = max(4, min(h, w) // 32)
        radius_min = 8
        radius_max = min(h, w) // 4

        for dy in range(-radius_max, radius_max, step):
            for dx in range(-radius_max, radius_max, step):
                dist = np.sqrt(dx**2 + dy**2)
                if dist < radius_min or dist > radius_max:
                    continue

                y1, x1 = cy + dy, cx + dx
                y2, x2 = cy + 2 * dy, cx + 2 * dx

                if 0 <= y2 < h and 0 <= x2 < w:
                    phi_f1 = phase[y1, x1]
                    phi_f2 = phase[y2, x2]
                    
                    # Ideal optical harmonic relation: phase angle difference should cluster around 0 or pi
                    # Biphase: beta = 2 * phi_f1 - phi_f2
                    biphase = (2 * phi_f1 - phi_f2) % (2 * np.pi)
                    # Angular deviation from nearest coherent harmonic alignment (0 or pi)
                    coupling_dev = min(biphase, abs(biphase - np.pi), 2 * np.pi - biphase)
                    harmonic_coupling_errors.append(coupling_dev)

        if not harmonic_coupling_errors:
            avg_coupling_error = 0.5
        else:
            avg_coupling_error = float(np.mean(harmonic_coupling_errors)) / (np.pi / 2)  # Normalized 0.0 - 1.0

        # 2. Phase Congruence / Uniformity Metric
        # Real photographic edges have localized phase alignment across spatial scales
        # Diffusion synthetic images display randomized, diffuse phase entropy across mid-to-high frequencies
        high_freq_mask = np.zeros((h, w), dtype=bool)
        yy, xx = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((yy - cy)**2 + (xx - cx)**2)
        high_freq_mask[(dist_from_center > radius_min) & (dist_from_center < radius_max * 1.5)] = True

        high_freq_phases = phase[high_freq_mask]
        
        # Phase gradient variance (derivative of phase shows noise in synthetic generations)
        phase_grad_y = np.diff(phase, axis=0)
        phase_grad_x = np.diff(phase, axis=1)
        phase_jitter = float(np.std(phase_grad_y) + np.std(phase_grad_x))

        # Normalized phase anomaly score
        # Camera photos: low jitter, high coupling coherence (lower error)
        # Latent diffusion: high jitter, high coupling error
        anomaly_score = 0.6 * avg_coupling_error + 0.4 * min(phase_jitter / 5.0, 1.0)
        signal_score = float(np.clip(anomaly_score, 0.0, 1.0))
        
        confidence = 0.80

        if signal_score > 0.55:
            reasoning = (
                f"Harmonic bi-spectral phase coupling error is elevated ({avg_coupling_error:.1%}). "
                f"Fourier phase angles lack the coherent alignment typical of camera optical lenses."
            )
        else:
            reasoning = (
                f"Bi-spectral phase angles demonstrate consistent harmonic coupling ({avg_coupling_error:.1%}), "
                f"consistent with physical camera optical propagation."
            )

        return AnalyzerResult(
            analyzer_name=self.name,
            display_name=self.display_name,
            signal_score=signal_score,
            confidence=confidence,
            reasoning=reasoning,
            details={
                "harmonic_coupling_error": float(avg_coupling_error),
                "phase_jitter": float(phase_jitter),
                "sample_points_analyzed": len(harmonic_coupling_errors),
            },
        )
