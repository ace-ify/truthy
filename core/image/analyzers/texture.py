"""
Texture Analyzer.
Uses Local Binary Patterns (LBP) and micro-texture statistics.
AI images often have subtly different texture characteristics than real photos.
"""
import numpy as np
from core.image.analyzers.base import BaseAnalyzer, AnalyzerResult
from core.image.preprocessor import ImageData


class TextureAnalyzer(BaseAnalyzer):

    name = "texture"
    display_name = "Texture Analysis"
    weight = 1.1

    def _analyze(self, image_data: ImageData) -> AnalyzerResult:
        gray = (image_data.grayscale * 255).astype(np.uint8)
        h, w = gray.shape

        # --- Signal 1: Local Binary Pattern histogram ---
        lbp = self._compute_lbp(gray)
        lbp_hist, _ = np.histogram(lbp.flatten(), bins=256, range=(0, 256), density=True)

        # LBP entropy: real photos have higher texture entropy
        lbp_entropy = -np.sum(lbp_hist[lbp_hist > 0] * np.log2(lbp_hist[lbp_hist > 0]))
        max_entropy = np.log2(256)
        normalized_entropy = lbp_entropy / max_entropy

        # --- Signal 2: Texture homogeneity ---
        # AI images sometimes have patches that are "too smooth" or "too uniform"
        block_size = 32
        block_entropies = []
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = lbp[y:y + block_size, x:x + block_size]
                bh, _ = np.histogram(block.flatten(), bins=64, range=(0, 256), density=True)
                be = -np.sum(bh[bh > 0] * np.log2(bh[bh > 0]))
                block_entropies.append(be)

        block_entropies = np.array(block_entropies) if block_entropies else np.array([lbp_entropy])
        texture_variance = np.var(block_entropies)
        texture_mean = np.mean(block_entropies)

        # --- Signal 3: Texture repetition detection ---
        # AI can produce subtle repeating patterns
        # Check autocorrelation of LBP map at multiple offsets
        lbp_float = lbp.astype(np.float64) / 255.0
        lbp_mean = lbp_float.mean()
        lbp_var = np.var(lbp_float)

        repetition_scores = []
        offsets = [(16, 0), (0, 16), (32, 0), (0, 32), (16, 16)]
        for dy, dx in offsets:
            if dy < h and dx < w:
                region1 = lbp_float[:h - max(dy, 1), :w - max(dx, 1)]
                region2 = lbp_float[dy:h - max(0, 1 - dy), dx:w - max(0, 1 - dx)]
                min_h = min(region1.shape[0], region2.shape[0])
                min_w = min(region1.shape[1], region2.shape[1])
                r1 = region1[:min_h, :min_w]
                r2 = region2[:min_h, :min_w]
                if lbp_var > 1e-10:
                    corr = np.mean((r1 - lbp_mean) * (r2 - lbp_mean)) / lbp_var
                    repetition_scores.append(abs(corr))

        max_repetition = max(repetition_scores) if repetition_scores else 0.0

        # --- Signal 4: GLCM-like texture contrast ---
        # Compute contrast from co-occurrence (simplified)
        # Horizontal neighbor differences
        h_diff = np.abs(gray[:, 1:].astype(np.float64) - gray[:, :-1].astype(np.float64))
        v_diff = np.abs(gray[1:, :].astype(np.float64) - gray[:-1, :].astype(np.float64))
        mean_contrast = (h_diff.mean() + v_diff.mean()) / 2
        contrast_std = (h_diff.std() + v_diff.std()) / 2

        # --- Combine signals ---
        # Very low entropy = too smooth/uniform texture = suspicious
        if normalized_entropy < 0.5:
            entropy_score = (0.5 - normalized_entropy) * 2
        elif normalized_entropy > 0.9:
            entropy_score = 0.2  # Also somewhat suspicious — too chaotic
        else:
            entropy_score = 0.0

        # High texture repetition = suspicious
        repetition_score = min(max_repetition * 3, 1.0) if max_repetition > 0.2 else 0.0

        # Very low texture variance = too uniform across image = suspicious
        if texture_mean > 0:
            cv = np.sqrt(texture_variance) / texture_mean  # Coefficient of variation
        else:
            cv = 0.0
        uniformity_score = max(0.3 - cv, 0) * 3 if cv < 0.3 else 0.0

        # Very low contrast = too smooth
        contrast_score = max(5.0 - mean_contrast, 0) / 5.0 if mean_contrast < 5.0 else 0.0

        signal_score = (
            entropy_score * 0.30
            + repetition_score * 0.25
            + uniformity_score * 0.25
            + contrast_score * 0.20
        )
        signal_score = np.clip(signal_score, 0.0, 1.0)

        confidence = 0.45 + 0.25 * min(min(h, w) / 512, 1.0)

        reasons = []
        if entropy_score > 0.3:
            reasons.append("abnormal texture entropy distribution")
        if repetition_score > 0.3:
            reasons.append("repeating texture patterns detected")
        if uniformity_score > 0.3:
            reasons.append("overly uniform texture across regions")
        if contrast_score > 0.3:
            reasons.append("unusually low micro-texture contrast")

        if signal_score > 0.5:
            reasoning = "Texture analysis shows AI indicators: " + "; ".join(reasons) if reasons else \
                "Micro-texture patterns differ from natural photography"
        else:
            reasoning = "Texture patterns are consistent with natural image capture"

        return AnalyzerResult(
            analyzer_name=self.name,
            display_name=self.display_name,
            signal_score=float(signal_score),
            confidence=float(confidence),
            reasoning=reasoning,
            details={
                "lbp_entropy": float(lbp_entropy),
                "normalized_entropy": float(normalized_entropy),
                "texture_variance": float(texture_variance),
                "max_repetition": float(max_repetition),
                "mean_contrast": float(mean_contrast),
                "entropy_score": float(entropy_score),
                "repetition_score": float(repetition_score),
                "uniformity_score": float(uniformity_score),
                "contrast_score": float(contrast_score),
            },
        )

    @staticmethod
    def _compute_lbp(gray: np.ndarray, radius: int = 1) -> np.ndarray:
        """Compute basic LBP (8-neighbor, given radius)."""
        h, w = gray.shape
        lbp = np.zeros((h, w), dtype=np.uint8)

        # 8 neighbors at given radius
        offsets = [
            (-radius, -radius), (-radius, 0), (-radius, radius),
            (0, radius), (radius, radius), (radius, 0),
            (radius, -radius), (0, -radius),
        ]

        padded = np.pad(gray, radius, mode='edge')

        for bit, (dy, dx) in enumerate(offsets):
            neighbor = padded[radius + dy:radius + dy + h, radius + dx:radius + dx + w]
            lbp |= ((neighbor >= gray).astype(np.uint8) << bit)

        return lbp
