"""
Statistical Analyzer.
Benford's law on pixel values, chi-square LSB analysis, histogram shape.
These detect statistical anomalies that differ between real and AI images.
"""
import numpy as np
from core.image.analyzers.base import BaseAnalyzer, AnalyzerResult
from core.image.preprocessor import ImageData


class StatisticalAnalyzer(BaseAnalyzer):

    name = "statistical"
    display_name = "Statistical Analysis"
    weight = 0.8

    def _analyze(self, image_data: ImageData) -> AnalyzerResult:
        img = image_data.np_array  # uint8 HxWx3
        gray = image_data.grayscale  # float64 0-1
        h, w = gray.shape

        # --- Signal 1: Benford's law analysis ---
        # First digit distribution of pixel values should follow Benford's law
        # for natural images (leading digit frequency)
        benford_expected = np.array([
            np.log10(1 + 1 / d) for d in range(1, 10)
        ])

        # Extract first significant digits from pixel values
        flat_pixels = img.flatten().astype(np.int32)
        nonzero = flat_pixels[flat_pixels > 0]
        if len(nonzero) > 100:
            first_digits = []
            for val in nonzero:
                s = str(val)
                first_digits.append(int(s[0]))

            digit_counts = np.zeros(9)
            for d in first_digits:
                if 1 <= d <= 9:
                    digit_counts[d - 1] += 1

            digit_freq = digit_counts / digit_counts.sum()

            # Chi-square-like divergence from Benford's
            benford_divergence = np.sum(
                (digit_freq - benford_expected) ** 2 / (benford_expected + 1e-10)
            )
        else:
            benford_divergence = 0.0
            digit_freq = np.zeros(9)

        # --- Signal 2: LSB (Least Significant Bit) analysis ---
        # In natural images, LSBs have specific statistical properties
        # AI images may have more uniform or patterned LSBs
        lsb = img & 1  # Extract LSB plane
        lsb_flat = lsb.flatten()
        lsb_mean = lsb_flat.mean()
        # Should be close to 0.5 for natural randomness
        lsb_bias = abs(lsb_mean - 0.5)

        # Chi-square test on LSB pairs
        lsb_r = img[:, :, 0] & 1
        lsb_g = img[:, :, 1] & 1
        lsb_b = img[:, :, 2] & 1

        # Check correlation between channel LSBs
        rg_lsb_corr = abs(_pearson_flat(lsb_r.flatten().astype(float), lsb_g.flatten().astype(float)))
        gb_lsb_corr = abs(_pearson_flat(lsb_g.flatten().astype(float), lsb_b.flatten().astype(float)))
        avg_lsb_corr = (rg_lsb_corr + gb_lsb_corr) / 2

        # --- Signal 3: Histogram smoothness ---
        # Real images: smooth, natural histogram. AI: can have gaps or unusual shapes
        hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 1))
        hist_norm = hist / hist.sum()

        # Compute histogram "bumpiness" (second derivative energy)
        if len(hist_norm) > 2:
            hist_2nd_deriv = np.diff(hist_norm, n=2)
            hist_roughness = np.sum(hist_2nd_deriv ** 2)
        else:
            hist_roughness = 0.0

        # Count empty bins (gaps in histogram)
        empty_bins = np.sum(hist == 0)
        empty_ratio = empty_bins / 256

        # --- Signal 4: Pixel value entropy ---
        pixel_entropy = -np.sum(hist_norm[hist_norm > 0] * np.log2(hist_norm[hist_norm > 0]))
        max_entropy = np.log2(256)
        norm_entropy = pixel_entropy / max_entropy

        # --- Signal 5: Adjacent pixel correlation ---
        # Natural images have high adjacent pixel correlation
        # AI might have slightly different correlation patterns
        flat_gray = (gray * 255).astype(np.float64)
        if w > 1:
            h_corr = _pearson_flat(flat_gray[:, :-1].flatten(), flat_gray[:, 1:].flatten())
        else:
            h_corr = 0.9
        if h > 1:
            v_corr = _pearson_flat(flat_gray[:-1, :].flatten(), flat_gray[1:, :].flatten())
        else:
            v_corr = 0.9
        adj_corr = (h_corr + v_corr) / 2

        # --- Combine signals ---
        # High Benford divergence = suspicious
        benford_score = min(benford_divergence / 0.5, 1.0) if benford_divergence > 0.05 else 0.0

        # High LSB bias = suspicious
        lsb_score = min(lsb_bias * 10, 1.0) if lsb_bias > 0.02 else 0.0

        # High LSB correlation = suspicious (LSBs should be ~independent)
        lsb_corr_score = min(avg_lsb_corr * 3, 1.0) if avg_lsb_corr > 0.1 else 0.0

        # Very smooth or very rough histogram
        if hist_roughness < 1e-7:
            roughness_score = 0.4  # Suspiciously smooth
        elif hist_roughness > 0.001:
            roughness_score = 0.3  # Very bumpy
        else:
            roughness_score = 0.0

        # Many empty histogram bins
        empty_score = min(empty_ratio * 3, 1.0) if empty_ratio > 0.1 else 0.0

        # Low adjacent pixel correlation is unusual
        adj_corr_score = max(0.9 - adj_corr, 0) / 0.2 if adj_corr < 0.9 else 0.0

        signal_score = (
            benford_score * 0.20
            + lsb_score * 0.15
            + lsb_corr_score * 0.15
            + roughness_score * 0.15
            + empty_score * 0.15
            + adj_corr_score * 0.20
        )
        signal_score = np.clip(signal_score, 0.0, 1.0)

        confidence = 0.40 + 0.20 * min(min(h, w) / 512, 1.0)

        reasons = []
        if benford_score > 0.3:
            reasons.append("pixel value distribution deviates from Benford's law")
        if lsb_score > 0.3:
            reasons.append("biased least-significant-bit distribution")
        if lsb_corr_score > 0.3:
            reasons.append("unusual LSB correlation between color channels")
        if roughness_score > 0.2:
            reasons.append("abnormal histogram shape")
        if empty_score > 0.3:
            reasons.append("gaps in pixel value histogram")
        if adj_corr_score > 0.3:
            reasons.append("low adjacent pixel correlation")

        if signal_score > 0.5:
            reasoning = "Statistical analysis shows anomalies: " + "; ".join(reasons) if reasons else \
                "Pixel statistics differ from natural image patterns"
        else:
            reasoning = "Statistical properties are consistent with natural photography"

        return AnalyzerResult(
            analyzer_name=self.name,
            display_name=self.display_name,
            signal_score=float(signal_score),
            confidence=float(confidence),
            reasoning=reasoning,
            details={
                "benford_divergence": float(benford_divergence),
                "lsb_bias": float(lsb_bias),
                "avg_lsb_correlation": float(avg_lsb_corr),
                "histogram_roughness": float(hist_roughness),
                "empty_bin_ratio": float(empty_ratio),
                "pixel_entropy": float(pixel_entropy),
                "adjacent_correlation": float(adj_corr),
            },
        )


def _pearson_flat(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    a_m, b_m = a.mean(), b.mean()
    a_s, b_s = a.std(), b.std()
    if a_s < 1e-10 or b_s < 1e-10:
        return 0.0
    return float(np.mean((a - a_m) * (b - b_m)) / (a_s * b_s))
