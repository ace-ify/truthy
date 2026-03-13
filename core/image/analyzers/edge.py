"""
Edge/Gradient Analyzer.
Analyzes edge characteristics, gradient distributions, and Laplacian variance.
AI images have different edge profiles than camera-captured photos.
"""
import numpy as np
from scipy.ndimage import sobel, laplace, gaussian_filter
from core.image.analyzers.base import BaseAnalyzer, AnalyzerResult
from core.image.preprocessor import ImageData


class EdgeAnalyzer(BaseAnalyzer):

    name = "edge"
    display_name = "Edge Analysis"
    weight = 1.0

    def _analyze(self, image_data: ImageData) -> AnalyzerResult:
        gray = image_data.grayscale  # float64, 0-1
        h, w = gray.shape

        # --- Signal 1: Gradient magnitude distribution ---
        grad_x = sobel(gray, axis=1)
        grad_y = sobel(gray, axis=0)
        grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

        grad_mean = grad_magnitude.mean()
        grad_std = grad_magnitude.std()
        grad_max = grad_magnitude.max()

        # Real images: gradient distribution follows a heavy-tailed distribution
        # AI images may have more uniform/truncated gradient distributions
        grad_flat = grad_magnitude.flatten()
        if grad_std > 1e-10:
            grad_standardized = (grad_flat - grad_mean) / grad_std
            grad_kurtosis = np.mean(grad_standardized ** 4) - 3
            grad_skewness = np.mean(grad_standardized ** 3)
        else:
            grad_kurtosis = 0.0
            grad_skewness = 0.0

        # --- Signal 2: Laplacian variance (focus/sharpness measure) ---
        lap = laplace(gray)
        lap_var = np.var(lap)
        lap_mean = np.mean(np.abs(lap))

        # --- Signal 3: Edge direction consistency ---
        # Real photos have diverse edge directions; AI can be too consistent
        grad_direction = np.arctan2(grad_y, grad_x)
        # Only consider pixels with significant gradient
        edge_threshold = np.percentile(grad_magnitude, 70)
        strong_edges = grad_magnitude > edge_threshold

        if strong_edges.any():
            strong_directions = grad_direction[strong_edges]
            # Histogram of edge directions (circular)
            dir_hist, _ = np.histogram(strong_directions, bins=36, range=(-np.pi, np.pi), density=True)
            dir_entropy = -np.sum(dir_hist[dir_hist > 0] * np.log2(dir_hist[dir_hist > 0]))
            max_dir_entropy = np.log2(36)
            dir_uniformity = dir_entropy / max_dir_entropy  # 1.0 = uniform, <1.0 = biased
        else:
            dir_uniformity = 0.5

        # --- Signal 4: Edge sharpness profile ---
        # AI images often have either too-sharp or unnaturally smooth edges
        # Compare edge profiles at different blur levels
        blurred_1 = gaussian_filter(gray, sigma=1.0)
        blurred_2 = gaussian_filter(gray, sigma=2.0)
        grad_sharp = np.sqrt(sobel(blurred_1, 0) ** 2 + sobel(blurred_1, 1) ** 2)
        grad_soft = np.sqrt(sobel(blurred_2, 0) ** 2 + sobel(blurred_2, 1) ** 2)

        if grad_soft.mean() > 1e-10:
            sharpness_ratio = grad_sharp.mean() / grad_soft.mean()
        else:
            sharpness_ratio = 1.0

        # --- Signal 5: Block-wise edge density variation ---
        block_size = 64
        edge_densities = []
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = grad_magnitude[y:y + block_size, x:x + block_size]
                density = np.mean(block > edge_threshold)
                edge_densities.append(density)

        edge_densities = np.array(edge_densities) if edge_densities else np.array([0.3])
        edge_density_cv = np.std(edge_densities) / (np.mean(edge_densities) + 1e-10)

        # --- Combine signals ---
        # Low gradient kurtosis = less heavy-tailed = suspicious (AI often smoother)
        kurtosis_score = max(3.0 - grad_kurtosis, 0) / 6.0 if grad_kurtosis < 3.0 else 0.0

        # Unusual sharpness ratio (too sharp or too soft)
        if sharpness_ratio > 2.5:
            sharpness_score = min((sharpness_ratio - 2.5) / 2.0, 1.0)
        elif sharpness_ratio < 1.2:
            sharpness_score = (1.2 - sharpness_ratio) * 2
        else:
            sharpness_score = 0.0

        # Very uniform edge directions = suspicious
        if dir_uniformity > 0.92:
            direction_score = (dir_uniformity - 0.92) / 0.08
        elif dir_uniformity < 0.5:
            direction_score = (0.5 - dir_uniformity)
        else:
            direction_score = 0.0

        # Low edge density variation = too uniform = suspicious
        density_score = max(0.3 - edge_density_cv, 0) / 0.3 if edge_density_cv < 0.3 else 0.0

        # Very low Laplacian variance = very smooth = suspicious
        lap_score = max(0.001 - lap_var, 0) / 0.001 if lap_var < 0.001 else 0.0

        signal_score = (
            kurtosis_score * 0.25
            + sharpness_score * 0.25
            + direction_score * 0.20
            + density_score * 0.15
            + lap_score * 0.15
        )
        signal_score = np.clip(signal_score, 0.0, 1.0)

        confidence = 0.40 + 0.25 * min(min(h, w) / 512, 1.0)

        reasons = []
        if kurtosis_score > 0.3:
            reasons.append("gradient distribution lacks natural heavy-tail characteristics")
        if sharpness_score > 0.3:
            reasons.append("abnormal edge sharpness profile")
        if direction_score > 0.3:
            reasons.append("unusual edge direction distribution")
        if density_score > 0.3:
            reasons.append("overly uniform edge density across image")
        if lap_score > 0.3:
            reasons.append("very low image detail complexity")

        if signal_score > 0.5:
            reasoning = "Edge analysis shows AI indicators: " + "; ".join(reasons) if reasons else \
                "Edge characteristics differ from natural photography"
        else:
            reasoning = "Edge and gradient patterns are consistent with real image capture"

        return AnalyzerResult(
            analyzer_name=self.name,
            display_name=self.display_name,
            signal_score=float(signal_score),
            confidence=float(confidence),
            reasoning=reasoning,
            details={
                "grad_mean": float(grad_mean),
                "grad_kurtosis": float(grad_kurtosis),
                "grad_skewness": float(grad_skewness),
                "laplacian_variance": float(lap_var),
                "sharpness_ratio": float(sharpness_ratio),
                "direction_uniformity": float(dir_uniformity),
                "edge_density_cv": float(edge_density_cv),
            },
        )
