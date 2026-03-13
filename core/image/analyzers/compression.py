"""
JPEG Compression Analyzer.
Analyzes DCT coefficient distributions, double compression artifacts,
and JPEG grid alignment consistency.
"""
import numpy as np
import io
from PIL import Image
from core.image.analyzers.base import BaseAnalyzer, AnalyzerResult
from core.image.preprocessor import ImageData


class CompressionAnalyzer(BaseAnalyzer):

    name = "compression"
    display_name = "Compression Analysis"
    weight = 0.8

    def _analyze(self, image_data: ImageData) -> AnalyzerResult:
        gray = image_data.grayscale
        h, w = gray.shape
        is_jpeg = image_data.original_format.upper() in ("JPEG", "JPG")

        # --- Signal 1: JPEG quality estimation ---
        # Re-save at multiple qualities and find closest match
        pil = image_data.pil_image
        estimated_quality = self._estimate_jpeg_quality(pil)

        # --- Signal 2: DCT coefficient analysis ---
        # Compute 8x8 DCT blocks and analyze coefficient distributions
        gray_uint8 = (gray * 255).astype(np.float64)

        # Pad to multiple of 8
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h or pad_w:
            gray_padded = np.pad(gray_uint8, ((0, pad_h), (0, pad_w)), mode='edge')
        else:
            gray_padded = gray_uint8

        ph, pw = gray_padded.shape
        dct_coeffs = []

        from scipy.fft import dctn
        for y in range(0, ph, 8):
            for x in range(0, pw, 8):
                block = gray_padded[y:y + 8, x:x + 8]
                dct_block = dctn(block - 128, type=2, norm='ortho')
                # Collect non-DC coefficients
                dct_coeffs.extend(dct_block.flatten()[1:])  # Skip DC

        dct_coeffs = np.array(dct_coeffs)

        if len(dct_coeffs) > 0:
            dct_mean = np.mean(np.abs(dct_coeffs))
            dct_std = np.std(dct_coeffs)
            # Kurtosis of DCT coefficients
            if dct_std > 1e-10:
                dct_kurtosis = np.mean(((dct_coeffs - dct_coeffs.mean()) / dct_std) ** 4) - 3
            else:
                dct_kurtosis = 0.0

            # Zero-crossing ratio (percentage of near-zero coefficients)
            near_zero = np.sum(np.abs(dct_coeffs) < 0.5) / len(dct_coeffs)
        else:
            dct_mean = 0
            dct_std = 0
            dct_kurtosis = 0
            near_zero = 0

        # --- Signal 3: Double compression detection ---
        # Compare error levels at two different quality levels
        if is_jpeg:
            err_q85 = self._compression_error(pil, 85)
            err_q95 = self._compression_error(pil, 95)
            double_comp_ratio = err_q85 / (err_q95 + 1e-10)
        else:
            err_q85 = 0
            err_q95 = 0
            double_comp_ratio = 1.0

        # --- Signal 4: JPEG grid alignment ---
        # Check for 8x8 block boundary artifacts
        if is_jpeg:
            # Measure discontinuity at block boundaries vs within blocks
            boundary_disc = 0
            interior_disc = 0
            count_b = 0
            count_i = 0

            for y in range(1, min(h, 256)):
                for x in range(1, min(w, 256)):
                    diff = abs(float(gray_uint8[y, x]) - float(gray_uint8[y - 1, x]))
                    if y % 8 == 0:
                        boundary_disc += diff
                        count_b += 1
                    else:
                        interior_disc += diff
                        count_i += 1

            if count_b > 0 and count_i > 0:
                grid_ratio = (boundary_disc / count_b) / (interior_disc / count_i + 1e-10)
            else:
                grid_ratio = 1.0
        else:
            grid_ratio = 1.0

        # --- Combine signals ---
        # Unusual quality level (too perfect or very specific)
        if estimated_quality >= 98:
            quality_score = 0.3  # Suspiciously high quality
        elif estimated_quality <= 50:
            quality_score = 0.2  # Very low quality — might be trying to hide artifacts
        else:
            quality_score = 0.0

        # Unusual DCT kurtosis
        # Real JPEG images have specific DCT coefficient distributions
        if is_jpeg:
            if dct_kurtosis < 5:
                dct_score = min((5 - dct_kurtosis) / 10, 0.5)
            else:
                dct_score = 0.0
        else:
            # For non-JPEG, check if DCT coefficients look "too clean"
            dct_score = max(0.7 - near_zero, 0) if near_zero < 0.7 else 0.0

        # Double compression signal
        if is_jpeg and double_comp_ratio > 1.5:
            double_score = min((double_comp_ratio - 1.5) / 3, 0.5)
        else:
            double_score = 0.0

        # Grid alignment issues
        if is_jpeg and abs(grid_ratio - 1.0) > 0.3:
            grid_score = min(abs(grid_ratio - 1.0) / 2, 0.5)
        else:
            grid_score = 0.0

        signal_score = (
            quality_score * 0.20
            + dct_score * 0.35
            + double_score * 0.25
            + grid_score * 0.20
        )
        signal_score = np.clip(signal_score, 0.0, 1.0)

        confidence = 0.55 if is_jpeg else 0.25

        reasons = []
        if quality_score > 0.2:
            reasons.append(f"unusual JPEG quality ({estimated_quality}%)")
        if dct_score > 0.2:
            reasons.append("DCT coefficient distribution atypical")
        if double_score > 0.2:
            reasons.append("possible double compression detected")
        if grid_score > 0.2:
            reasons.append("JPEG grid alignment anomalies")

        if signal_score > 0.5:
            reasoning = "Compression analysis shows anomalies: " + "; ".join(reasons) if reasons else \
                "Compression artifacts are inconsistent with typical camera output"
        else:
            reasoning = "Compression characteristics are consistent with standard image processing"

        return AnalyzerResult(
            analyzer_name=self.name,
            display_name=self.display_name,
            signal_score=float(signal_score),
            confidence=float(confidence),
            reasoning=reasoning,
            details={
                "is_jpeg": is_jpeg,
                "estimated_quality": int(estimated_quality),
                "dct_mean": float(dct_mean),
                "dct_kurtosis": float(dct_kurtosis),
                "near_zero_ratio": float(near_zero),
                "double_comp_ratio": float(double_comp_ratio),
                "grid_ratio": float(grid_ratio),
            },
        )

    @staticmethod
    def _estimate_jpeg_quality(pil_image: Image.Image) -> int:
        """Estimate JPEG quality by binary search on re-save error."""
        original = np.array(pil_image, dtype=np.float64)
        best_q = 90
        best_err = float('inf')

        for q in [60, 70, 80, 85, 90, 95, 98]:
            buf = io.BytesIO()
            pil_image.save(buf, format="JPEG", quality=q)
            buf.seek(0)
            resaved = np.array(Image.open(buf), dtype=np.float64)
            if resaved.shape == original.shape:
                err = np.mean(np.abs(original - resaved))
                if err < best_err:
                    best_err = err
                    best_q = q
        return best_q

    @staticmethod
    def _compression_error(pil_image: Image.Image, quality: int) -> float:
        """Compute mean absolute error from re-saving at given quality."""
        original = np.array(pil_image, dtype=np.float64)
        buf = io.BytesIO()
        pil_image.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        resaved = np.array(Image.open(buf), dtype=np.float64)
        if resaved.shape == original.shape:
            return float(np.mean(np.abs(original - resaved)))
        return 0.0
