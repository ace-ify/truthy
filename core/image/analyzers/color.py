"""
Color Channel Analyzer.
Analyzes RGB channel correlations, histogram distributions, and chrominance patterns.
AI generators produce subtly different color statistics than camera sensors.
"""
import numpy as np
from core.image.analyzers.base import BaseAnalyzer, AnalyzerResult
from core.image.preprocessor import ImageData


class ColorAnalyzer(BaseAnalyzer):

    name = "color"
    display_name = "Color Analysis"
    weight = 0.3

    def _analyze(self, image_data: ImageData) -> AnalyzerResult:
        img = image_data.np_array.astype(np.float64) / 255.0
        h, w, _ = img.shape
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

        # --- Signal 1: Inter-channel correlation ---
        # Real camera images: RGB channels are highly correlated (same scene)
        # AI images may have slightly different correlation structure
        rg_corr = _pearson(r.flatten(), g.flatten())
        gb_corr = _pearson(g.flatten(), b.flatten())
        rb_corr = _pearson(r.flatten(), b.flatten())
        avg_corr = (abs(rg_corr) + abs(gb_corr) + abs(rb_corr)) / 3

        # --- Signal 2: Channel histogram shape similarity ---
        r_hist, _ = np.histogram(r.flatten(), bins=64, range=(0, 1), density=True)
        g_hist, _ = np.histogram(g.flatten(), bins=64, range=(0, 1), density=True)
        b_hist, _ = np.histogram(b.flatten(), bins=64, range=(0, 1), density=True)

        # Bhattacharyya distance between channel histograms
        rg_dist = _bhattacharyya(r_hist, g_hist)
        gb_dist = _bhattacharyya(g_hist, b_hist)
        rb_dist = _bhattacharyya(r_hist, b_hist)
        avg_dist = (rg_dist + gb_dist + rb_dist) / 3

        # --- Signal 3: Color saturation distribution ---
        # Convert to HSV-like saturation
        c_max = np.maximum(np.maximum(r, g), b)
        c_min = np.minimum(np.minimum(r, g), b)
        saturation = np.where(c_max > 0, (c_max - c_min) / (c_max + 1e-10), 0)

        sat_mean = saturation.mean()
        sat_std = saturation.std()
        sat_hist, _ = np.histogram(saturation.flatten(), bins=32, range=(0, 1), density=True)
        sat_entropy = -np.sum(sat_hist[sat_hist > 0] * np.log2(sat_hist[sat_hist > 0]))

        # --- Signal 4: Color gamut coverage ---
        # AI images sometimes use limited or unusual color gamut
        # Sample color space coverage using 3D histogram
        color_hist, _ = np.histogramdd(
            img.reshape(-1, 3),
            bins=8,
            range=[(0, 1), (0, 1), (0, 1)],
        )
        total_bins = 8 ** 3
        occupied_bins = np.sum(color_hist > 0)
        gamut_coverage = occupied_bins / total_bins

        # --- Signal 5: Block-wise color consistency ---
        # Check if color statistics vary naturally across image
        block_size = 64
        block_saturations = []
        block_hues = []
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = img[y:y + block_size, x:x + block_size]
                br, bg, bb = block[:, :, 0], block[:, :, 1], block[:, :, 2]
                bc_max = np.maximum(np.maximum(br, bg), bb)
                bc_min = np.minimum(np.minimum(br, bg), bb)
                bsat = np.where(bc_max > 0, (bc_max - bc_min) / (bc_max + 1e-10), 0)
                block_saturations.append(bsat.mean())

        block_saturations = np.array(block_saturations) if block_saturations else np.array([sat_mean])
        sat_cv = np.std(block_saturations) / (np.mean(block_saturations) + 1e-10)

        # --- Signal 6: Unusual value distributions ---
        # Check for values clustering at exact boundaries (0, 255) — clipping artifacts
        r_uint = image_data.np_array[:, :, 0]
        g_uint = image_data.np_array[:, :, 1]
        b_uint = image_data.np_array[:, :, 2]
        total_pixels = h * w
        clip_ratio = (
            np.sum((r_uint == 0) | (r_uint == 255))
            + np.sum((g_uint == 0) | (g_uint == 255))
            + np.sum((b_uint == 0) | (b_uint == 255))
        ) / (3 * total_pixels)

        # --- Combine signals ---
        # Very high channel correlation is actually normal for real photos
        # Low correlation is suspicious
        corr_score = max(0.85 - avg_corr, 0) / 0.35 if avg_corr < 0.85 else 0.0

        # Large inter-channel histogram distance is slightly suspicious
        dist_score = min(avg_dist * 3, 1.0) if avg_dist > 0.1 else 0.0

        # Unusual saturation distribution
        if sat_mean < 0.05:
            sat_score = 0.3  # Very desaturated — could be AI
        elif sat_mean > 0.6:
            sat_score = 0.3  # Hyper-saturated — could be AI
        else:
            sat_score = 0.0

        # Low gamut coverage = limited palette = slightly suspicious
        gamut_score = max(0.15 - gamut_coverage, 0) / 0.15 if gamut_coverage < 0.15 else 0.0

        # Very low saturation variation = too uniform = suspicious
        sat_var_score = max(0.2 - sat_cv, 0) / 0.2 if sat_cv < 0.2 else 0.0

        signal_score = (
            corr_score * 0.25
            + dist_score * 0.20
            + sat_score * 0.15
            + gamut_score * 0.15
            + sat_var_score * 0.15
            + min(clip_ratio * 5, 0.3) * 0.10  # Clip ratio signal
        )
        signal_score = np.clip(signal_score, 0.0, 1.0)

        confidence = 0.35 + 0.25 * min(min(h, w) / 512, 1.0)

        reasons = []
        if corr_score > 0.3:
            reasons.append("low inter-channel correlation")
        if dist_score > 0.3:
            reasons.append("divergent channel histogram shapes")
        if sat_score > 0.2:
            reasons.append("unusual color saturation distribution")
        if gamut_score > 0.3:
            reasons.append("limited color gamut usage")
        if sat_var_score > 0.3:
            reasons.append("overly uniform color saturation across regions")

        if signal_score > 0.5:
            reasoning = "Color analysis shows AI indicators: " + "; ".join(reasons) if reasons else \
                "Color characteristics differ from typical camera output"
        else:
            reasoning = "Color channel properties are consistent with natural photography"

        return AnalyzerResult(
            analyzer_name=self.name,
            display_name=self.display_name,
            signal_score=float(signal_score),
            confidence=float(confidence),
            reasoning=reasoning,
            details={
                "rg_correlation": float(rg_corr),
                "gb_correlation": float(gb_corr),
                "rb_correlation": float(rb_corr),
                "avg_correlation": float(avg_corr),
                "avg_hist_distance": float(avg_dist),
                "saturation_mean": float(sat_mean),
                "saturation_std": float(sat_std),
                "gamut_coverage": float(gamut_coverage),
                "clip_ratio": float(clip_ratio),
            },
        )


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a_m, b_m = a.mean(), b.mean()
    a_s, b_s = a.std(), b.std()
    if a_s < 1e-10 or b_s < 1e-10:
        return 0.0
    return float(np.mean((a - a_m) * (b - b_m)) / (a_s * b_s))


def _bhattacharyya(h1: np.ndarray, h2: np.ndarray) -> float:
    """Bhattacharyya distance between two normalized histograms."""
    h1_n = h1 / (h1.sum() + 1e-10)
    h2_n = h2 / (h2.sum() + 1e-10)
    bc = np.sum(np.sqrt(h1_n * h2_n))
    return float(-np.log(bc + 1e-10)) if bc > 0 else 10.0
