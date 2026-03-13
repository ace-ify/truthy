"""
Frequency/FFT Analyzer.
Detects spectral artifacts left by AI image generators (especially GANs).
AI-generated images often show periodic peaks in the Fourier domain.
"""
import numpy as np
from core.image.analyzers.base import BaseAnalyzer, AnalyzerResult
from core.image.preprocessor import ImageData


class FrequencyAnalyzer(BaseAnalyzer):

    name = "frequency"
    display_name = "Frequency Analysis"
    weight = 1.5  # High value signal

    def _analyze(self, image_data: ImageData) -> AnalyzerResult:
        gray = image_data.grayscale
        h, w = gray.shape

        # 2D FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.log1p(np.abs(f_shift))

        # Normalize magnitude spectrum
        mag_norm = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-10)

        # --- Signal 1: Radial energy distribution ---
        # Real images have smooth radial falloff; GANs have periodic bumps
        cy, cx = h // 2, w // 2
        max_radius = min(cy, cx)
        radial_profile = []
        for r in range(1, max_radius):
            # Create ring mask
            y, x = np.ogrid[-cy:h - cy, -cx:w - cx]
            ring = (x * x + y * y >= (r - 1) ** 2) & (x * x + y * y < r ** 2)
            if ring.any():
                radial_profile.append(mag_norm[ring].mean())

        radial_profile = np.array(radial_profile)

        # Measure radial smoothness (second derivative = roughness)
        if len(radial_profile) > 4:
            second_deriv = np.diff(radial_profile, n=2)
            roughness = np.std(second_deriv)
        else:
            roughness = 0.0

        # --- Signal 2: High-frequency energy ratio ---
        # AI images often have unusual high-frequency content
        hf_radius = max_radius * 0.7
        y, x = np.ogrid[-cy:h - cy, -cx:w - cx]
        hf_mask = (x * x + y * y) >= hf_radius ** 2
        lf_mask = ~hf_mask

        hf_energy = mag_norm[hf_mask].mean() if hf_mask.any() else 0
        lf_energy = mag_norm[lf_mask].mean() if lf_mask.any() else 1
        hf_ratio = hf_energy / (lf_energy + 1e-10)

        # --- Signal 3: Spectral peak detection ---
        # GANs leave periodic peaks; count anomalous peaks
        if len(radial_profile) > 10:
            median_profile = np.median(radial_profile)
            std_profile = np.std(radial_profile)
            threshold = median_profile + 2.5 * std_profile
            peak_count = int(np.sum(radial_profile > threshold))
            peak_ratio = peak_count / len(radial_profile)
        else:
            peak_count = 0
            peak_ratio = 0.0

        # --- Signal 4: Azimuthal symmetry ---
        # GAN artifacts often create grid-like patterns (high azimuthal symmetry)
        angular_bins = 36
        angular_energies = []
        for i in range(angular_bins):
            angle_start = (2 * np.pi * i) / angular_bins
            angle_end = (2 * np.pi * (i + 1)) / angular_bins
            y_grid, x_grid = np.ogrid[-cy:h - cy, -cx:w - cx]
            angles = np.arctan2(y_grid, x_grid) % (2 * np.pi)
            angle_mask = (angles >= angle_start) & (angles < angle_end)
            mid_mask = (x_grid * x_grid + y_grid * y_grid > (max_radius * 0.2) ** 2) & \
                       (x_grid * x_grid + y_grid * y_grid < (max_radius * 0.8) ** 2)
            combined = angle_mask & mid_mask
            if combined.any():
                angular_energies.append(mag_norm[combined].mean())

        if angular_energies:
            angular_std = np.std(angular_energies)
            angular_uniformity = 1.0 - min(angular_std * 10, 1.0)
        else:
            angular_uniformity = 0.5

        # --- Combine signals into score ---
        # Higher roughness = more likely AI (periodic artifacts)
        roughness_score = min(roughness * 50, 1.0)
        # Unusual HF ratio
        hf_score = min(abs(hf_ratio - 0.15) * 5, 1.0) if hf_ratio > 0.25 else 0.0
        # Spectral peaks
        peak_score = min(peak_ratio * 20, 1.0)
        # High angular uniformity = suspicious (grid artifacts)
        symmetry_score = angular_uniformity if angular_uniformity > 0.7 else 0.0

        signal_score = (
            roughness_score * 0.30
            + hf_score * 0.25
            + peak_score * 0.25
            + symmetry_score * 0.20
        )
        signal_score = np.clip(signal_score, 0.0, 1.0)

        # Confidence based on image size (larger = more reliable frequency analysis)
        size_factor = min(min(h, w) / 256, 1.0)
        confidence = 0.4 + 0.4 * size_factor  # 0.4-0.8 range

        # Reasoning
        reasons = []
        if roughness_score > 0.5:
            reasons.append("irregular spectral roughness (periodic artifacts)")
        if hf_score > 0.5:
            reasons.append("unusual high-frequency energy distribution")
        if peak_score > 0.5:
            reasons.append(f"{peak_count} anomalous spectral peaks detected")
        if symmetry_score > 0.5:
            reasons.append("high angular symmetry (grid-like artifacts)")

        if signal_score > 0.5:
            reasoning = "Spectral analysis shows AI indicators: " + "; ".join(reasons) if reasons else \
                "Frequency spectrum shows subtle AI generation artifacts"
        else:
            reasoning = "Frequency spectrum appears consistent with natural photography"

        return AnalyzerResult(
            analyzer_name=self.name,
            display_name=self.display_name,
            signal_score=float(signal_score),
            confidence=float(confidence),
            reasoning=reasoning,
            details={
                "roughness": float(roughness),
                "roughness_score": float(roughness_score),
                "hf_ratio": float(hf_ratio),
                "hf_score": float(hf_score),
                "peak_count": peak_count,
                "peak_score": float(peak_score),
                "angular_uniformity": float(angular_uniformity),
                "symmetry_score": float(symmetry_score),
            },
        )
