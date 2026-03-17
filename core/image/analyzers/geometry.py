"""
Geometry Analyzer.
Checks shadow consistency, perspective plausibility, and reflection physics.
Most useful for photos with clear geometric cues (buildings, people, objects).
"""
import numpy as np
from scipy.ndimage import sobel, gaussian_filter, label
from core.image.analyzers.base import BaseAnalyzer, AnalyzerResult
from core.image.preprocessor import ImageData


class GeometryAnalyzer(BaseAnalyzer):

    name = "geometry"
    display_name = "Geometry Analysis"
    weight = 0.3  # Lower weight — not always applicable

    def _analyze(self, image_data: ImageData) -> AnalyzerResult:
        gray = image_data.grayscale
        img = image_data.np_array.astype(np.float64) / 255.0
        h, w = gray.shape

        # --- Signal 1: Shadow direction consistency ---
        # Compute gradient direction in dark regions (shadows)
        grad_x = sobel(gray, axis=1)
        grad_y = sobel(gray, axis=0)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
        grad_dir = np.arctan2(grad_y, grad_x)

        # Identify shadow regions (dark areas with nearby brighter areas)
        blurred = gaussian_filter(gray, sigma=5)
        shadow_mask = (gray < 0.3) & (blurred > 0.35) & (grad_mag > 0.02)

        if shadow_mask.sum() > 100:
            shadow_dirs = grad_dir[shadow_mask]
            # Check if shadow directions are consistent (they should be if single light source)
            dir_hist, _ = np.histogram(shadow_dirs, bins=36, range=(-np.pi, np.pi))
            dir_hist_norm = dir_hist / (dir_hist.sum() + 1e-10)
            shadow_entropy = -np.sum(dir_hist_norm[dir_hist_norm > 0] * np.log2(dir_hist_norm[dir_hist_norm > 0]))
            max_entropy = np.log2(36)
            shadow_consistency = 1.0 - shadow_entropy / max_entropy  # Higher = more consistent

            # Find dominant shadow direction
            dominant_bin = np.argmax(dir_hist)
            dominant_angle = -np.pi + (2 * np.pi * dominant_bin / 36)
        else:
            shadow_consistency = 0.5  # Can't determine
            dominant_angle = 0.0

        # --- Signal 2: Perspective line analysis ---
        # Detect strong horizontal/vertical edges and check convergence
        strong_threshold = np.percentile(grad_mag, 90)
        strong_mask = grad_mag > strong_threshold

        if strong_mask.sum() > 50:
            strong_dirs = grad_dir[strong_mask]
            # Count edges in cardinal directions (±10 degrees of 0, 90, 180, 270)
            cardinal = 0
            total = len(strong_dirs)
            for angle_center in [0, np.pi / 2, -np.pi / 2, np.pi, -np.pi]:
                cardinal += np.sum(np.abs(strong_dirs - angle_center) < np.radians(10))

            cardinal_ratio = cardinal / total if total > 0 else 0
        else:
            cardinal_ratio = 0.5

        # --- Signal 3: Symmetry analysis ---
        # AI images are sometimes more symmetric than natural photos
        left_half = gray[:, :w // 2]
        right_half = np.fliplr(gray[:, w - w // 2:])
        min_w_half = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_w_half]
        right_half = right_half[:, :min_w_half]

        if left_half.size > 0 and right_half.size > 0:
            lr_diff = np.mean(np.abs(left_half - right_half))
            symmetry = 1.0 - min(lr_diff * 5, 1.0)  # Higher = more symmetric
        else:
            symmetry = 0.5

        # Top/bottom symmetry
        top_half = gray[:h // 2, :]
        bottom_half = np.flipud(gray[h - h // 2:, :])
        min_h_half = min(top_half.shape[0], bottom_half.shape[0])
        top_half = top_half[:min_h_half, :]
        bottom_half = bottom_half[:min_h_half, :]

        if top_half.size > 0 and bottom_half.size > 0:
            tb_diff = np.mean(np.abs(top_half - bottom_half))
            tb_symmetry = 1.0 - min(tb_diff * 5, 1.0)
        else:
            tb_symmetry = 0.5

        # --- Signal 4: Brightness gradient plausibility ---
        # Natural scenes have plausible light falloff; AI may have impossible lighting
        # Check if brightness gradient is spatially smooth
        brightness_blocks = []
        block_size = max(min(h, w) // 8, 16)
        for y in range(0, h - block_size, block_size):
            row = []
            for x in range(0, w - block_size, block_size):
                block = gray[y:y + block_size, x:x + block_size]
                row.append(block.mean())
            brightness_blocks.append(row)

        if brightness_blocks and len(brightness_blocks[0]) > 1:
            brightness_grid = np.array(brightness_blocks)
            # Compute second derivative of brightness grid
            if brightness_grid.shape[0] > 2 and brightness_grid.shape[1] > 2:
                bx = np.diff(brightness_grid, axis=1)
                by = np.diff(brightness_grid, axis=0)
                bxx = np.diff(bx, axis=1) if bx.shape[1] > 1 else np.array([0])
                byy = np.diff(by, axis=0) if by.shape[0] > 1 else np.array([0])
                lighting_roughness = (np.std(bxx) + np.std(byy)) / 2
            else:
                lighting_roughness = 0.0
        else:
            lighting_roughness = 0.0

        # --- Combine signals ---
        # Very low shadow consistency (random shadow directions) = suspicious
        if shadow_mask.sum() > 100:
            shadow_score = max(0.4 - shadow_consistency, 0) / 0.4 if shadow_consistency < 0.4 else 0.0
        else:
            shadow_score = 0.0

        # Very high symmetry = suspicious (natural photos rarely very symmetric)
        symmetry_score = 0.0
        if symmetry > 0.85:
            symmetry_score = (symmetry - 0.85) / 0.15
        if tb_symmetry > 0.85:
            symmetry_score = max(symmetry_score, (tb_symmetry - 0.85) / 0.15)

        # Very high cardinal ratio = suspiciously geometric
        perspective_score = max(cardinal_ratio - 0.5, 0) if cardinal_ratio > 0.5 else 0.0

        # High lighting roughness = inconsistent lighting
        lighting_score = min(lighting_roughness * 20, 1.0) if lighting_roughness > 0.01 else 0.0

        signal_score = (
            shadow_score * 0.30
            + symmetry_score * 0.30
            + perspective_score * 0.15
            + lighting_score * 0.25
        )
        signal_score = np.clip(signal_score, 0.0, 1.0)

        # Confidence depends on whether geometric cues were available
        has_geometric = shadow_mask.sum() > 200 or cardinal_ratio > 0.3
        confidence = 0.45 if has_geometric else 0.25

        reasons = []
        if shadow_score > 0.3:
            reasons.append("inconsistent shadow directions")
        if symmetry_score > 0.3:
            reasons.append("unnaturally high image symmetry")
        if lighting_score > 0.3:
            reasons.append("abrupt lighting inconsistencies")
        if perspective_score > 0.3:
            reasons.append("unusual perspective line distribution")

        if signal_score > 0.5:
            reasoning = "Geometry analysis shows anomalies: " + "; ".join(reasons) if reasons else \
                "Geometric properties differ from natural photography"
        else:
            reasoning = "Shadow, perspective, and lighting appear physically plausible"

        return AnalyzerResult(
            analyzer_name=self.name,
            display_name=self.display_name,
            signal_score=float(signal_score),
            confidence=float(confidence),
            reasoning=reasoning,
            details={
                "shadow_consistency": float(shadow_consistency),
                "shadow_pixels": int(shadow_mask.sum()),
                "lr_symmetry": float(symmetry),
                "tb_symmetry": float(tb_symmetry),
                "cardinal_ratio": float(cardinal_ratio),
                "lighting_roughness": float(lighting_roughness),
            },
        )
