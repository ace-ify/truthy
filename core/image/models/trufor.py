"""
Forensic Heatmap Generator (TruFor-inspired).
Combines per-pixel ELA, noise residual, and frequency analysis
into a spatial manipulation probability heatmap.
Returns heatmap as base64 PNG for frontend visualization.
"""
import numpy as np
import io
import base64
import logging
from PIL import Image
from scipy.ndimage import gaussian_filter, sobel, uniform_filter
from scipy.fft import fft2, fftshift

from core.image.analyzers.base import BaseAnalyzer, AnalyzerResult
from core.image.preprocessor import ImageData

logger = logging.getLogger(__name__)


class TruForDetector(BaseAnalyzer):
    """Generates a forensic manipulation heatmap from multiple signals."""

    name = "trufor_heatmap"
    display_name = "Forensic Heatmap"
    weight = 0.8  # Low weight — this is a heuristic combo, not the real TruFor model

    # ELA resave quality
    ELA_QUALITY = 90
    # Block size for local analysis
    BLOCK_SIZE = 16

    def _analyze(self, image_data: ImageData) -> AnalyzerResult:
        gray = image_data.grayscale  # float64, 0-1
        img = image_data.np_array  # uint8 HxWxC
        pil = image_data.pil_image
        h, w = gray.shape
        bs = self.BLOCK_SIZE

        # --- Layer 1: ELA heatmap ---
        ela_map = self._compute_ela_map(pil, h, w)

        # --- Layer 2: Noise inconsistency map ---
        noise_map = self._compute_noise_map(gray, h, w, bs)

        # --- Layer 3: Frequency anomaly map ---
        freq_map = self._compute_freq_map(gray, h, w, bs)

        # --- Layer 4: Edge sharpness map ---
        edge_map = self._compute_edge_map(gray)

        # --- Combine into final heatmap ---
        combined = (
            ela_map * 0.30
            + noise_map * 0.30
            + freq_map * 0.20
            + edge_map * 0.20
        )
        combined = np.clip(combined, 0, 1)

        # Smooth for visual appeal
        combined = gaussian_filter(combined, sigma=3)
        combined = np.clip(combined, 0, 1)

        # --- Compute overall score from heatmap ---
        mean_heat = float(np.mean(combined))
        max_heat = float(np.max(combined))
        high_area = float(np.mean(combined > 0.5))  # Fraction of image with high score

        # Score: large areas of high heat = suspicious
        if high_area > 0.3:
            signal_score = min(0.5 + high_area, 1.0)
        elif mean_heat > 0.4:
            signal_score = mean_heat * 1.2
        else:
            signal_score = mean_heat

        signal_score = float(np.clip(signal_score, 0, 1))
        confidence = 0.50

        # --- Generate heatmap image ---
        heatmap_b64 = self._render_heatmap(combined)

        reasons = []
        if high_area > 0.3:
            reasons.append(f"{high_area:.0%} of image shows high manipulation probability")
        if mean_heat > 0.4:
            reasons.append("elevated forensic anomaly levels across the image")

        if signal_score > 0.5:
            reasoning = "Forensic heatmap shows significant anomalies: " + "; ".join(reasons) if reasons else \
                "Pixel-level forensics suggest manipulation or AI generation"
        else:
            reasoning = "Forensic heatmap shows consistent, natural pixel-level properties"

        return AnalyzerResult(
            analyzer_name=self.name,
            display_name=self.display_name,
            signal_score=signal_score,
            confidence=confidence,
            reasoning=reasoning,
            details={
                "mean_heatmap_value": float(mean_heat),
                "max_heatmap_value": float(max_heat),
                "high_area_ratio": float(high_area),
                "heatmap_b64": heatmap_b64,
            },
        )

    def _compute_ela_map(self, pil: Image.Image, h: int, w: int) -> np.ndarray:
        """Pixel-level Error Level Analysis."""
        original = np.array(pil.convert("RGB"), dtype=np.float64)

        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=self.ELA_QUALITY)
        buf.seek(0)
        resaved = np.array(Image.open(buf).convert("RGB"), dtype=np.float64)

        if resaved.shape != original.shape:
            return np.zeros((h, w))

        ela = np.mean(np.abs(original - resaved), axis=2)
        # Normalize to 0-1
        ela_max = np.percentile(ela, 99) if np.max(ela) > 0 else 1.0
        ela_norm = np.clip(ela / (ela_max + 1e-10), 0, 1)

        # Invert: uniform ELA = suspicious, varied = natural
        # Compute local variance of ELA
        ela_local_mean = uniform_filter(ela_norm, size=32)
        ela_local_sq = uniform_filter(ela_norm ** 2, size=32)
        ela_local_var = np.clip(ela_local_sq - ela_local_mean ** 2, 0, None)

        # Low local variance = suspiciously uniform
        uniformity = 1.0 - np.clip(np.sqrt(ela_local_var) * 10, 0, 1)
        return uniformity

    def _compute_noise_map(self, gray: np.ndarray, h: int, w: int, bs: int) -> np.ndarray:
        """Block-wise noise consistency analysis."""
        noise = gray - gaussian_filter(gray, sigma=2)
        noise_map = np.zeros((h, w))

        # Compute reference noise variance
        noise_vars = []
        for y in range(0, h - bs, bs):
            for x in range(0, w - bs, bs):
                block = noise[y:y + bs, x:x + bs]
                noise_vars.append(np.var(block))

        if not noise_vars:
            return noise_map

        median_var = np.median(noise_vars)
        if median_var < 1e-10:
            return noise_map

        # Mark blocks that deviate from median noise level
        idx = 0
        for y in range(0, h - bs, bs):
            for x in range(0, w - bs, bs):
                if idx < len(noise_vars):
                    ratio = abs(noise_vars[idx] - median_var) / (median_var + 1e-10)
                    score = min(ratio / 3, 1.0)
                    noise_map[y:y + bs, x:x + bs] = score
                    idx += 1

        return gaussian_filter(noise_map, sigma=bs / 2)

    def _compute_freq_map(self, gray: np.ndarray, h: int, w: int, bs: int) -> np.ndarray:
        """Block-wise frequency content analysis."""
        freq_map = np.zeros((h, w))

        # Compute per-block high-frequency ratio
        hf_ratios = []
        coords = []
        for y in range(0, h - bs, bs):
            for x in range(0, w - bs, bs):
                block = gray[y:y + bs, x:x + bs]
                f = np.abs(fft2(block))
                total_energy = np.sum(f) + 1e-10
                # High-frequency = outer half of FFT
                cy, cx = bs // 2, bs // 2
                yy, xx = np.ogrid[:bs, :bs]
                hf_mask = ((yy - cy) ** 2 + (xx - cx) ** 2) > (min(bs, bs) // 4) ** 2
                hf_energy = np.sum(f[hf_mask])
                ratio = hf_energy / total_energy
                hf_ratios.append(ratio)
                coords.append((y, x))

        if not hf_ratios:
            return freq_map

        median_hf = np.median(hf_ratios)
        if median_hf < 1e-10:
            return freq_map

        for (y, x), ratio in zip(coords, hf_ratios):
            deviation = abs(ratio - median_hf) / (median_hf + 1e-10)
            score = min(deviation / 2, 1.0)
            freq_map[y:y + bs, x:x + bs] = score

        return gaussian_filter(freq_map, sigma=bs / 2)

    def _compute_edge_map(self, gray: np.ndarray) -> np.ndarray:
        """Edge sharpness anomaly map."""
        gx = sobel(gray, axis=1)
        gy = sobel(gray, axis=0)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)

        # Compute local edge sharpness ratio at two scales
        sharp_fine = gaussian_filter(grad_mag, sigma=1)
        sharp_coarse = gaussian_filter(grad_mag, sigma=5)

        # Ratio: if fine == coarse, edges are blurred (natural)
        # If fine >> coarse, edges are artificially sharp
        ratio = np.where(
            sharp_coarse > 1e-10,
            sharp_fine / (sharp_coarse + 1e-10),
            0
        )
        # Normalize: values near 1 = natural, much higher = artificial
        edge_anomaly = np.clip((ratio - 1.0) / 3.0, 0, 1)
        return gaussian_filter(edge_anomaly, sigma=3)

    def _render_heatmap(self, heatmap: np.ndarray) -> str:
        """Convert heatmap array to a base64-encoded colored PNG."""
        h, w = heatmap.shape

        # Vectorized blue→yellow→red colormap
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        low = heatmap < 0.5
        high = ~low

        # Blue to Yellow for values < 0.5
        t_low = heatmap[low] * 2
        colored[low, 0] = (255 * t_low).astype(np.uint8)
        colored[low, 1] = (255 * t_low).astype(np.uint8)
        colored[low, 2] = (255 * (1 - t_low)).astype(np.uint8)

        # Yellow to Red for values >= 0.5
        t_high = (heatmap[high] - 0.5) * 2
        colored[high, 0] = 255
        colored[high, 1] = (255 * (1 - t_high)).astype(np.uint8)
        colored[high, 2] = 0

        pil_heatmap = Image.fromarray(colored)
        if max(h, w) > 512:
            pil_heatmap.thumbnail((512, 512), Image.NEAREST)

        buf = io.BytesIO()
        pil_heatmap.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
