"""
CLIP-based AI image detector using zero-shot classification.
Uses CLIP's vision-language alignment to compare image against
"real photograph" vs "AI-generated image" prompts.
Falls back to HuggingFace Inference API when available.
"""
import numpy as np
import requests
import os
import io
import time
from pathlib import Path
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from config import CLIP_MODEL_ID

from core.image.analyzers.base import BaseAnalyzer, AnalyzerResult
from core.image.preprocessor import ImageData


# Prompt sets for zero-shot classification
REAL_PROMPTS = [
    "a real photograph taken by a camera",
    "an authentic unedited photograph",
    "a natural photograph of a real scene",
]

AI_PROMPTS = [
    "an AI-generated image",
    "a digitally synthesized artificial image",
    "an image created by artificial intelligence",
]


class CLIPDetector(BaseAnalyzer):
    """CLIP-based zero-shot AI image detector."""

    name = "clip_detector"
    display_name = "CLIP Zero-Shot"
    weight = 0.4  # Low weight — zero-shot classification is unreliable for AI detection

    HF_ZS_API_URL = "https://router.huggingface.co/hf-inference/models/openai/clip-vit-large-patch14"

    def __init__(self):
        self.hf_token = os.environ.get("HF_TOKEN", "")
        self.use_api = bool(self.hf_token)
        self.model = None
        self.processor = None

    def _analyze(self, image_data: ImageData) -> AnalyzerResult:
        if self.use_api:
            result = self._predict_api(image_data)
            if result.error is None:
                return result
        # Fallback to local
        return self._predict_local(image_data)

    def _predict_api(self, image_data: ImageData) -> AnalyzerResult:
        """Use HuggingFace zero-shot image classification API."""
        headers = {"Authorization": f"Bearer {self.hf_token}"}

        # Convert image to JPEG bytes
        buffer = io.BytesIO()
        pil = image_data.pil_image.copy()
        if max(pil.size) > 1024:
            pil.thumbnail((1024, 1024), Image.LANCZOS)
        pil.save(buffer, format="JPEG", quality=90)
        image_bytes = buffer.getvalue()

        candidate_labels = [
            "a real photograph",
            "an AI-generated image",
        ]

        import json
        payload = {
            "parameters": {"candidate_labels": candidate_labels}
        }

        max_retries = 2
        for attempt in range(max_retries):
            try:
                # Use zero-shot image classification endpoint
                response = requests.post(
                    self.HF_ZS_API_URL,
                    headers={
                        "Authorization": f"Bearer {self.hf_token}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "inputs": {
                            "image": _image_to_base64(image_bytes),
                        },
                        "parameters": {
                            "candidate_labels": candidate_labels,
                        },
                    },
                    timeout=30,
                )

                if response.status_code == 200:
                    results = response.json()
                    return self._parse_zs_response(results, candidate_labels)

                elif response.status_code == 503:
                    time.sleep(10)
                    continue
                else:
                    break

            except requests.exceptions.Timeout:
                continue
            except Exception as e:
                return AnalyzerResult.failed(self.name, self.display_name, f"API error: {e}")

        return AnalyzerResult.failed(self.name, self.display_name, "API unavailable")

    def _parse_zs_response(self, results: dict, candidate_labels: list) -> AnalyzerResult:
        """Parse zero-shot classification response."""
        scores = {}
        if isinstance(results, list):
            for item in results:
                scores[item.get("label", "")] = item.get("score", 0.0)
        elif isinstance(results, dict):
            labels = results.get("labels", [])
            probs = results.get("scores", [])
            for label, prob in zip(labels, probs):
                scores[label] = prob

        ai_score = scores.get("an AI-generated image", 0.5)
        real_score = scores.get("a real photograph", 0.5)

        signal_score = float(ai_score)
        # Dynamic confidence: how decisive is the split?
        spread = abs(ai_score - real_score)
        confidence = 0.40 + 0.35 * min(spread / 0.8, 1.0)  # 0.40-0.75 range

        if signal_score > 0.5:
            reasoning = f"CLIP classifies image as AI-generated ({signal_score:.0%} vs {real_score:.0%} real)"
        else:
            reasoning = f"CLIP classifies image as real photograph ({real_score:.0%} vs {ai_score:.0%} AI)"

        return AnalyzerResult(
            analyzer_name=self.name,
            display_name=self.display_name,
            signal_score=signal_score,
            confidence=confidence,
            reasoning=reasoning,
            details={
                "model": CLIP_MODEL_ID,
                "ai_score": float(ai_score),
                "real_score": float(real_score),
                "method": "hf_api_zeroshot",
            },
        )

    def _predict_local(self, image_data: ImageData) -> AnalyzerResult:
        """Local inference using CLIP for zero-shot classification."""
        try:
            if self.model is None:
                self._load_local_model()

            import torch

            pil = image_data.pil_image.copy()
            if max(pil.size) > 1024:
                pil.thumbnail((1024, 1024), Image.LANCZOS)

            all_prompts = REAL_PROMPTS + AI_PROMPTS
            inputs = self.processor(
                text=all_prompts,
                images=pil,
                return_tensors="pt",
                padding=True,
            )

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)[0]

            probs_np = probs.numpy()
            real_prob = float(probs_np[:len(REAL_PROMPTS)].sum())
            ai_prob = float(probs_np[len(REAL_PROMPTS):].sum())

            # Normalize
            total = real_prob + ai_prob
            if total > 0:
                real_prob /= total
                ai_prob /= total

            signal_score = ai_prob
            # Dynamic confidence from local zero-shot
            spread = abs(ai_prob - real_prob)
            confidence = 0.35 + 0.30 * min(spread / 0.8, 1.0)  # 0.35-0.65 range

            if signal_score > 0.5:
                reasoning = f"CLIP zero-shot classifies as AI-generated ({signal_score:.0%})"
            else:
                reasoning = f"CLIP zero-shot classifies as real photograph ({1 - signal_score:.0%})"

            return AnalyzerResult(
                analyzer_name=self.name,
                display_name=self.display_name,
                signal_score=float(signal_score),
                confidence=float(confidence),
                reasoning=reasoning,
                details={
                    "model": CLIP_MODEL_ID,
                    "ai_probability": float(ai_prob),
                    "real_probability": float(real_prob),
                    "prompt_scores": {p: float(s) for p, s in zip(all_prompts, probs_np)},
                    "method": "local_zeroshot",
                },
            )

        except Exception as e:
            return AnalyzerResult.failed(self.name, self.display_name, f"Local inference failed: {e}")

    def _load_local_model(self):
        """Lazy-load CLIP model."""
        from transformers import CLIPModel, CLIPProcessor
        import gc

        gc.collect()
        self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
        self.model = CLIPModel.from_pretrained(
            CLIP_MODEL_ID,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        gc.collect()


def _image_to_base64(image_bytes: bytes) -> str:
    """Convert image bytes to base64 string."""
    import base64
    return base64.b64encode(image_bytes).decode("utf-8")
