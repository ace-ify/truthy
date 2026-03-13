"""
Noise Pattern Analyzer.
Real cameras leave sensor-specific noise (PRNU). AI images lack this.
Analyzes noise residual characteristics and spatial correlation.
"""
import numpy as np
from core.image.analyzers.base import BaseAnalyzer, AnalyzerResult
from core.image.preprocessor import ImageData

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from config import NOISE_BLOCK_SIZE


class NoiseAnalyzer(BaseAnalyzer):

    name = "noise"
    display_name = "Noise Pattern Analysis"
    weight = 1.4

    def _analyze(self, image_data: ImageData) -> AnalyzerResult:
        gray = image_data.grayscale  # float64, 0-1
        h, w = gray.shape

        # --- Extract noise residual ---
        # Denoise using simple Gaussian filter, then subtract
        from scipy.ndimage import gaussian_filter, uniform_filter
        denoised = gaussian_filter(gray, sigma=1.5)
        noise_residual = gray - denoised

        # --- Signal 1: Noise variance (overall noise level) ---
        noise_var = np.var(noise_residual)

        # --- Signal 2: Noise spatial correlation ---
        # Camera sensor noise has spatial correlation (PRNU pattern)
        # AI noise is typically more random (lower correlation)
        # Compute autocorrelation at lag 1 (horizontal)
        if w > 2:
            r1 = noise_residual[:, :-1]
            r2 = noise_residual[:, 1:]
            mean_r = noise_residual.mean()
            var_r = np.var(noise_residual)
            if var_r > 1e-10:
                autocorr_h = np.mean((r1 - mean_r) * (r2 - mean_r)) / var_r
            else:
                autocorr_h = 0.0
        else:
            autocorr_h = 0.0

        # Vertical autocorrelation
        if h > 2:
            r1 = noise_residual[:-1, :]
            r2 = noise_residual[1:, :]
            if var_r > 1e-10:
                autocorr_v = np.mean((r1 - mean_r) * (r2 - mean_r)) / var_r
            else:
                autocorr_v = 0.0
        else:
            autocorr_v = 0.0

        avg_autocorr = (autocorr_h + autocorr_v) / 2

        # --- Signal 3: Block-wise noise variance consistency ---
        # Real cameras: noise variance is fairly consistent across blocks
        # AI images: can have varying noise in different regions
        block_size = NOISE_BLOCK_SIZE
        block_vars = []
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = noise_residual[y:y + block_size, x:x + block_size]
                block_vars.append(np.var(block))

        block_vars = np.array(block_vars) if block_vars else np.array([noise_var])
        var_of_vars = np.var(block_vars)
        mean_of_vars = np.mean(block_vars)
        noise_consistency = 1.0 - min(var_of_vars / (mean_of_vars ** 2 + 1e-10), 1.0)

        # --- Signal 4: Channel noise correlation ---
        # Real camera noise is correlated across RGB channels (same sensor)
        # AI noise channels may be independent
        np_img = image_data.np_array.astype(np.float64) / 255.0
        channel_noises = []
        for c in range(3):
            ch = np_img[:, :, c]
            ch_denoised = gaussian_filter(ch, sigma=1.5)
            channel_noises.append(ch - ch_denoised)

        if len(channel_noises) == 3:
            # Correlation between R-G, G-B, R-B noise
            rg_corr = _pearson_2d(channel_noises[0], channel_noises[1])
            gb_corr = _pearson_2d(channel_noises[1], channel_noises[2])
            rb_corr = _pearson_2d(channel_noises[0], channel_noises[2])
            channel_correlation = (abs(rg_corr) + abs(gb_corr) + abs(rb_corr)) / 3
        else:
            channel_correlation = 0.5

        # --- Signal 5: Noise histogram normality ---
        # Camera noise is approximately Gaussian
        # AI noise may have different distributions
        noise_flat = noise_residual.flatten()
        noise_mean = np.mean(noise_flat)
        noise_std = np.std(noise_flat)
        if noise_std > 1e-10:
            standardized = (noise_flat - noise_mean) / noise_std
            kurtosis = np.mean(standardized ** 4) - 3  # Excess kurtosis (0 = Gaussian)
        else:
            kurtosis = 0.0

        # --- Combine signals ---
        # Very low noise = suspicious (AI can be "too clean")
        if noise_var < 1e-6:
            low_noise_score = 0.7
        elif noise_var < 5e-5:
            low_noise_score = 0.4
        else:
            low_noise_score = 0.0

        # Low spatial autocorrelation = no sensor pattern = suspicious
        autocorr_score = max(0.3 - avg_autocorr, 0) / 0.3 if avg_autocorr < 0.3 else 0.0

        # Low channel correlation = independent noise = suspicious
        channel_score = max(0.4 - channel_correlation, 0) / 0.4 if channel_correlation < 0.4 else 0.0

        # Non-Gaussian noise distribution
        kurtosis_score = min(abs(kurtosis) / 5.0, 1.0) if abs(kurtosis) > 1.0 else 0.0

        # Inconsistent block noise
        inconsistency_score = 1.0 - noise_consistency if noise_consistency < 0.7 else 0.0

        signal_score = (
            low_noise_score * 0.20
            + autocorr_score * 0.25
            + channel_score * 0.25
            + kurtosis_score * 0.15
            + inconsistency_score * 0.15
        )
        signal_score = np.clip(signal_score, 0.0, 1.0)

        # Confidence
        size_factor = min(min(h, w) / 256, 1.0)
        confidence = 0.40 + 0.35 * size_factor

        # Reasoning
        reasons = []
        if low_noise_score > 0.3:
            reasons.append("unusually low noise level (too clean)")
        if autocorr_score > 0.3:
            reasons.append("low spatial noise correlation (no camera sensor pattern)")
        if channel_score > 0.3:
            reasons.append("low cross-channel noise correlation")
        if kurtosis_score > 0.3:
            reasons.append("non-Gaussian noise distribution")
        if inconsistency_score > 0.3:
            reasons.append("inconsistent noise variance across regions")

        if signal_score > 0.5:
            reasoning = "Noise analysis suggests AI generation: " + "; ".join(reasons) if reasons else \
                "Noise patterns differ from real camera sensor characteristics"
        else:
            reasoning = "Noise patterns are consistent with real camera sensor capture"

        return AnalyzerResult(
            analyzer_name=self.name,
            display_name=self.display_name,
            signal_score=float(signal_score),
            confidence=float(confidence),
            reasoning=reasoning,
            details={
                "noise_variance": float(noise_var),
                "autocorr_horizontal": float(autocorr_h),
                "autocorr_vertical": float(autocorr_v),
                "avg_autocorrelation": float(avg_autocorr),
                "noise_consistency": float(noise_consistency),
                "channel_correlation": float(channel_correlation),
                "kurtosis": float(kurtosis),
                "low_noise_score": float(low_noise_score),
                "autocorr_score": float(autocorr_score),
                "channel_score": float(channel_score),
            },
        )


def _pearson_2d(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Pearson correlation between two 2D arrays."""
    a_flat = a.flatten()
    b_flat = b.flatten()
    a_mean = a_flat.mean()
    b_mean = b_flat.mean()
    a_std = a_flat.std()
    b_std = b_flat.std()
    if a_std < 1e-10 or b_std < 1e-10:
        return 0.0
    return float(np.mean((a_flat - a_mean) * (b_flat - b_mean)) / (a_std * b_std))
