"""
Error Level Analysis (ELA) Analyzer.
Re-saves the image as JPEG and compares pixel-level differences.
AI-generated images show different error patterns than real photos.
"""
import numpy as np
import io
from PIL import Image
from core.image.analyzers.base import BaseAnalyzer, AnalyzerResult
from core.image.preprocessor import ImageData

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from config import ELA_QUALITY


class ELAAnalyzer(BaseAnalyzer):

    name = "ela"
    display_name = "Error Level Analysis"
    weight = 1.3

    def _analyze(self, image_data: ImageData) -> AnalyzerResult:
        pil_image = image_data.pil_image
        w, h = pil_image.size

        # Re-save as JPEG at fixed quality
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=ELA_QUALITY)
        buffer.seek(0)
        resaved = Image.open(buffer)

        # Compute pixel-wise absolute difference
        original_arr = np.array(pil_image, dtype=np.float64)
        resaved_arr = np.array(resaved, dtype=np.float64)
        diff = np.abs(original_arr - resaved_arr)

        # ELA map (average across channels)
        ela_map = diff.mean(axis=2)

        # --- Signal 1: Overall ELA level ---
        mean_ela = ela_map.mean()
        max_ela = ela_map.max()

        # --- Signal 2: ELA uniformity ---
        # Real photos have varying ELA across regions; AI tends to be more uniform
        block_size = 64
        block_means = []
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = ela_map[y:y + block_size, x:x + block_size]
                block_means.append(block.mean())

        block_means = np.array(block_means) if block_means else np.array([mean_ela])
        uniformity = 1.0 - min(np.std(block_means) / (np.mean(block_means) + 1e-10), 1.0)

        # --- Signal 3: ELA distribution shape ---
        # Real photos: wide, skewed distribution. AI: narrow, more symmetric
        ela_flat = ela_map.flatten()
        ela_std = np.std(ela_flat)
        ela_median = np.median(ela_flat)

        # Skewness (simplified)
        skewness = np.mean(((ela_flat - mean_ela) / (ela_std + 1e-10)) ** 3)

        # Kurtosis (simplified) - AI images tend to have lower kurtosis
        kurtosis = np.mean(((ela_flat - mean_ela) / (ela_std + 1e-10)) ** 4) - 3

        # --- Signal 4: Edge-interior ELA difference ---
        # In real photos, edges have higher ELA than interiors
        # AI images often have more uniform ELA
        from scipy.ndimage import sobel
        edge_magnitude = np.sqrt(
            sobel(image_data.grayscale, axis=0) ** 2
            + sobel(image_data.grayscale, axis=1) ** 2
        )
        edge_mask = edge_magnitude > np.percentile(edge_magnitude, 75)
        interior_mask = ~edge_mask

        if edge_mask.any() and interior_mask.any():
            edge_ela = ela_map[edge_mask].mean()
            interior_ela = ela_map[interior_mask].mean()
            edge_contrast = (edge_ela - interior_ela) / (edge_ela + interior_ela + 1e-10)
        else:
            edge_contrast = 0.0

        # --- Combine signals ---
        # High uniformity = suspicious (AI-like)
        uniformity_score = max(uniformity - 0.5, 0) * 2  # 0.5-1.0 mapped to 0-1

        # Very low overall ELA = suspicious (already JPEG at similar quality)
        if mean_ela < 2.0:
            low_ela_score = 0.0  # Image might just be high quality
        elif mean_ela < 5.0:
            low_ela_score = 0.3
        else:
            low_ela_score = 0.0

        # Low edge contrast = suspicious (AI tends to have uniform ELA)
        contrast_score = max(0.5 - edge_contrast, 0) * 2 if edge_contrast < 0.5 else 0.0

        # Low kurtosis = suspicious
        kurtosis_score = max(0.5 - kurtosis / 10, 0) if kurtosis < 3 else 0.0

        signal_score = (
            uniformity_score * 0.35
            + contrast_score * 0.30
            + kurtosis_score * 0.20
            + low_ela_score * 0.15
        )
        signal_score = np.clip(signal_score, 0.0, 1.0)

        # Confidence: ELA is more reliable for JPEG images
        is_jpeg = image_data.original_format.upper() in ("JPEG", "JPG")
        confidence = 0.65 if is_jpeg else 0.40

        # Reasoning
        reasons = []
        if uniformity_score > 0.5:
            reasons.append("unusually uniform error levels across image regions")
        if contrast_score > 0.5:
            reasons.append("low edge-to-interior ELA contrast")
        if kurtosis_score > 0.5:
            reasons.append("ELA distribution shape inconsistent with natural photography")

        if signal_score > 0.5:
            reasoning = "ELA indicates potential AI generation: " + "; ".join(reasons) if reasons else \
                "Error level patterns differ from typical photographs"
        else:
            reasoning = "Error level analysis is consistent with authentic image capture"

        return AnalyzerResult(
            analyzer_name=self.name,
            display_name=self.display_name,
            signal_score=float(signal_score),
            confidence=float(confidence),
            reasoning=reasoning,
            details={
                "mean_ela": float(mean_ela),
                "max_ela": float(max_ela),
                "uniformity": float(uniformity),
                "uniformity_score": float(uniformity_score),
                "edge_contrast": float(edge_contrast),
                "contrast_score": float(contrast_score),
                "skewness": float(skewness),
                "kurtosis": float(kurtosis),
                "kurtosis_score": float(kurtosis_score),
                "is_jpeg": is_jpeg,
            },
        )
