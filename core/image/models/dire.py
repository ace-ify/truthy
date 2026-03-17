"""
DIRE — Diffusion Reconstruction Error detector.
Encodes an image through a pre-trained VAE (from Stable Diffusion),
decodes it, and compares the reconstruction to the original.
AI-generated images reconstruct with very low error because they
already live in the VAE's latent space. Real photos do not.
"""
import numpy as np
import io
import os
import time
import logging
from pathlib import Path
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from core.image.analyzers.base import BaseAnalyzer, AnalyzerResult
from core.image.preprocessor import ImageData

logger = logging.getLogger(__name__)

# The lightweight VAE from Stable Diffusion
VAE_MODEL_ID = "stabilityai/sd-vae-ft-mse"


class DIREDetector(BaseAnalyzer):
    """Diffusion Reconstruction Error detector."""

    name = "dire_detector"
    display_name = "DIRE Reconstruction"
    weight = 0.7  # DCT approximation is unreliable; only useful with real VAE

    def __init__(self):
        self.vae = None
        self.hf_token = os.environ.get("HF_TOKEN", "")

    def _analyze(self, image_data: ImageData) -> AnalyzerResult:
        # Try local VAE first (more accurate), then pixel-level approximation
        try:
            return self._predict_vae(image_data)
        except Exception as e:
            logger.debug(f"VAE inference unavailable ({e}), using DCT approximation")
            return self._predict_dct_approximation(image_data)

    def _predict_vae(self, image_data: ImageData) -> AnalyzerResult:
        """Use Stable Diffusion VAE to encode→decode and measure error."""
        if self.vae is None:
            self._load_vae()

        import torch
        from torchvision import transforms

        pil = image_data.pil_image.copy()
        # VAE expects 512x512, but we can handle other sizes
        target_size = 512
        pil_resized = pil.resize((target_size, target_size), Image.LANCZOS)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # Scale to [-1, 1]
        ])

        img_tensor = transform(pil_resized).unsqueeze(0)  # 1x3x512x512

        with torch.no_grad():
            # Encode to latent space
            latent = self.vae.encode(img_tensor).latent_dist.sample()
            # Decode back
            reconstructed = self.vae.decode(latent).sample

        # Convert back to [0, 1] range
        original_np = (img_tensor.squeeze().permute(1, 2, 0).numpy() + 1) / 2
        recon_np = (reconstructed.squeeze().permute(1, 2, 0).numpy() + 1) / 2
        recon_np = np.clip(recon_np, 0, 1)

        # Compute reconstruction metrics
        mse = float(np.mean((original_np - recon_np) ** 2))
        mae = float(np.mean(np.abs(original_np - recon_np)))

        # Per-channel error
        per_channel_mse = [
            float(np.mean((original_np[:, :, c] - recon_np[:, :, c]) ** 2))
            for c in range(3)
        ]

        # SSIM-like structural comparison (simplified)
        from scipy.ndimage import uniform_filter
        orig_gray = np.mean(original_np, axis=2)
        recon_gray = np.mean(recon_np, axis=2)

        mu_x = uniform_filter(orig_gray, size=11)
        mu_y = uniform_filter(recon_gray, size=11)
        sigma_x = uniform_filter(orig_gray ** 2, size=11) - mu_x ** 2
        sigma_y = uniform_filter(recon_gray ** 2, size=11) - mu_y ** 2
        sigma_xy = uniform_filter(orig_gray * recon_gray, size=11) - mu_x * mu_y

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
                   ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
        ssim = float(np.mean(ssim_map))

        # Spatial error distribution
        error_map = np.mean(np.abs(original_np - recon_np), axis=2)
        error_std = float(np.std(error_map))
        error_kurtosis = float(_kurtosis(error_map.flatten()))

        return self._score_from_metrics(
            mse, mae, ssim, per_channel_mse, error_std, error_kurtosis, "vae"
        )

    def _predict_dct_approximation(self, image_data: ImageData) -> AnalyzerResult:
        """
        Approximate DIRE using DCT-based compression-reconstruction.
        JPEG compression acts as a crude autoencoder: compress at low quality,
        decompress, and measure error. AI images tend to have lower error
        because their frequency content is smoother.
        """
        pil = image_data.pil_image.copy()
        if max(pil.size) > 512:
            pil.thumbnail((512, 512), Image.LANCZOS)

        original_np = np.array(pil, dtype=np.float64) / 255.0

        # "Encode-decode" via aggressive JPEG compression at multiple qualities
        errors = []
        for quality in [15, 25, 40]:
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            recon_pil = Image.open(buf).convert("RGB")
            recon_np = np.array(recon_pil, dtype=np.float64) / 255.0

            if recon_np.shape == original_np.shape:
                errors.append(np.mean(np.abs(original_np - recon_np)))

        if not errors:
            return AnalyzerResult.failed(self.name, self.display_name, "DCT approx failed")

        # Average reconstruction error across quality levels
        avg_error = float(np.mean(errors))
        error_variance = float(np.var(errors))

        # Also compute high-frequency content ratio
        gray = np.mean(original_np, axis=2)
        from scipy.fft import fft2, fftshift
        f = fftshift(fft2(gray))
        mag = np.abs(f)
        h, w = gray.shape
        cy, cx = h // 2, w // 2
        radius = min(h, w) // 4
        y, x = np.ogrid[:h, :w]
        high_freq_mask = ((y - cy) ** 2 + (x - cx) ** 2) > radius ** 2
        hf_ratio = float(np.sum(mag[high_freq_mask]) / (np.sum(mag) + 1e-10))

        # SSIM approximation using the q=40 reconstruction
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=40)
        buf.seek(0)
        recon40 = np.array(Image.open(buf).convert("RGB"), dtype=np.float64) / 255.0
        if recon40.shape == original_np.shape:
            mse = float(np.mean((original_np - recon40) ** 2))
            # Simple luminance-based SSIM approximation
            orig_gray = np.mean(original_np, axis=2)
            recon_gray = np.mean(recon40, axis=2)
            mu_o, mu_r = orig_gray.mean(), recon_gray.mean()
            sig_o, sig_r = orig_gray.std(), recon_gray.std()
            sig_or = np.mean((orig_gray - mu_o) * (recon_gray - mu_r))
            C1, C2 = 0.01 ** 2, 0.03 ** 2
            ssim = float(
                ((2 * mu_o * mu_r + C1) * (2 * sig_or + C2))
                / ((mu_o ** 2 + mu_r ** 2 + C1) * (sig_o ** 2 + sig_r ** 2 + C2))
            )
        else:
            mse = avg_error ** 2
            ssim = 0.5

        per_channel_mse = [0.0, 0.0, 0.0]
        error_map = np.mean(np.abs(original_np - recon40), axis=2) if recon40.shape == original_np.shape else np.array([0.0])
        error_std = float(np.std(error_map))
        error_kurtosis = float(_kurtosis(error_map.flatten()))

        return self._score_from_metrics(
            mse, avg_error, ssim, per_channel_mse, error_std, error_kurtosis,
            "dct_approximation",
            extra_details={
                "avg_reconstruction_error": avg_error,
                "error_variance": error_variance,
                "high_freq_ratio": hf_ratio,
            }
        )

    def _score_from_metrics(
        self, mse, mae, ssim, per_channel_mse, error_std, error_kurtosis,
        method, extra_details=None
    ) -> AnalyzerResult:
        """Convert reconstruction metrics into a signal score."""

        # Key insight: AI images have LOW reconstruction error (high SSIM)
        # Real photos have HIGH reconstruction error (low SSIM)

        # SSIM-based score: high SSIM = likely AI
        if ssim > 0.95:
            ssim_score = 0.9  # Very high reconstruction = very likely AI
        elif ssim > 0.90:
            ssim_score = 0.7
        elif ssim > 0.85:
            ssim_score = 0.4
        elif ssim > 0.75:
            ssim_score = 0.2
        else:
            ssim_score = 0.0  # Low SSIM = real photo

        # MSE-based score: low MSE = likely AI
        if mse < 0.001:
            mse_score = 0.9
        elif mse < 0.005:
            mse_score = 0.6
        elif mse < 0.01:
            mse_score = 0.3
        else:
            mse_score = 0.0

        # Error distribution: AI images have more uniform error
        # Low error_std = uniform error = suspicious
        if error_std < 0.02:
            dist_score = 0.6
        elif error_std < 0.04:
            dist_score = 0.3
        else:
            dist_score = 0.0

        # Combine
        signal_score = (
            ssim_score * 0.40
            + mse_score * 0.35
            + dist_score * 0.25
        )
        signal_score = float(np.clip(signal_score, 0.0, 1.0))

        # VAE method is true DIRE — high confidence
        # DCT approximation is a crude proxy — very low confidence so it doesn't distort the ensemble
        confidence = 0.55 if method == "vae" else 0.25

        if signal_score > 0.5:
            reasoning = f"Image reconstructs with low error (SSIM={ssim:.3f}), suggesting AI generation"
        else:
            reasoning = f"Image has high reconstruction error (SSIM={ssim:.3f}), consistent with real photography"

        details = {
            "mse": float(mse),
            "mae": float(mae),
            "ssim": float(ssim),
            "error_std": float(error_std),
            "error_kurtosis": float(error_kurtosis),
            "method": method,
        }
        if extra_details:
            details.update(extra_details)

        return AnalyzerResult(
            analyzer_name=self.name,
            display_name=self.display_name,
            signal_score=signal_score,
            confidence=confidence,
            reasoning=reasoning,
            details=details,
        )

    def _load_vae(self):
        """Lazy-load the Stable Diffusion VAE."""
        from diffusers import AutoencoderKL
        import gc

        gc.collect()
        self.vae = AutoencoderKL.from_pretrained(
            VAE_MODEL_ID,
            low_cpu_mem_usage=True,
        )
        self.vae.eval()
        gc.collect()


def _kurtosis(arr: np.ndarray) -> float:
    """Compute excess kurtosis."""
    m = arr.mean()
    s = arr.std()
    if s < 1e-10:
        return 0.0
    return float(np.mean(((arr - m) / s) ** 4) - 3)
