"""
Dual-CNN AI image detector.
Runs two complementary models:
  - dima806/deepfake_vs_real_image_detection: face-specific, excellent at confirming real photos
  - Ateeqq/ai-vs-human-image-detector: aggressive AI detector, excellent at catching AI images
dima806 prevents false positives; Ateeqq provides AI recall.
When they disagree, the judge uses ELA as a tiebreaker.
"""
import numpy as np
import os
import logging
from pathlib import Path
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from config import IMAGE_AI_THRESHOLD

from core.image.analyzers.base import BaseAnalyzer, AnalyzerResult
from core.image.preprocessor import ImageData

logger = logging.getLogger(__name__)

# Primary: face-specific deepfake detector (zero false positives on real photos)
FACE_MODEL = "dima806/deepfake_vs_real_image_detection"
# Secondary: aggressive AI detector (catches all AI but false-positive prone)
AI_MODEL = "Ateeqq/ai-vs-human-image-detector"


class CNNDetector(BaseAnalyzer):
    """Dual-CNN AI image detector — face-specific + aggressive AI models."""

    name = "cnn_detector"
    display_name = "CNN Classifier"
    weight = 2.5

    def __init__(self):
        self.hf_token = os.environ.get("HF_TOKEN", "")
        self.face_model = None
        self.face_proc = None
        self.ai_model = None
        self.ai_proc = None

    def _analyze(self, image_data: ImageData) -> AnalyzerResult:
        return self._predict_dual(image_data)

    def _predict_dual(self, image_data: ImageData) -> AnalyzerResult:
        """Run both models and combine scores."""
        try:
            if self.face_model is None:
                self._load_models()

            import torch

            pil = image_data.pil_image

            # Run whichever models are available
            face_score = (
                self._run_model(pil, self.face_proc, self.face_model)
                if self.face_model is not None
                else None
            )
            ai_score = (
                self._run_model(pil, self.ai_proc, self.ai_model)
                if self.ai_model is not None
                else None
            )

            # --- Combination logic ---
            if face_score is not None and ai_score is not None:
                # Both models agree AI → very strong signal
                if face_score > 0.5 and ai_score > 0.5:
                    signal_score = 0.5 * face_score + 0.5 * ai_score
                    confidence = 0.90
                    method = "both_ai"
                    reasoning = f"Both CNNs agree: AI-generated (face={face_score:.0%}, ai={ai_score:.0%})"

                # Both models agree Real → strong real signal
                elif face_score < 0.1 and ai_score < 0.5:
                    signal_score = 0.5 * face_score + 0.5 * ai_score
                    confidence = 0.85
                    method = "both_real"
                    reasoning = f"Both CNNs agree: human-created (face={face_score:.0%}, ai={ai_score:.0%})"

                # Face model says Real but AI model says AI → DISAGREEMENT
                elif face_score < 0.1 and ai_score > 0.5:
                    signal_score = 0.50  # Neutral — let the judge decide
                    confidence = 0.40   # Low confidence
                    method = "cnn_disagree"
                    reasoning = f"CNN models disagree: face={face_score:.0%} real, ai={ai_score:.0%} AI — needs tiebreaker"

                else:
                    signal_score = face_score
                    confidence = 0.35
                    method = "uncertain"
                    reasoning = f"CNN uncertain (face={face_score:.0%}, ai={ai_score:.0%}) — deferring to other analyzers"

            elif ai_score is not None:
                # Ateeqq is loaded
                signal_score = ai_score
                confidence = 0.85 if (ai_score > 0.8 or ai_score < 0.2) else 0.50
                method = "ai_model_only"
                status_str = "AI-generated" if ai_score > 0.5 else "human-created"
                reasoning = f"Vision Transformer detection: {status_str} ({ai_score:.0%} AI probability)"
                face_score = 0.5
            elif face_score is not None:
                # Face model only
                signal_score = face_score
                confidence = 0.80 if face_score < 0.1 else 0.40
                method = "face_model_only"
                status_str = "human-created" if face_score < 0.5 else "deepfake/AI"
                reasoning = f"Face model detection: {status_str} ({face_score:.0%} deepfake probability)"
                ai_score = 0.5
            else:
                return AnalyzerResult.failed(self.name, self.display_name, "No CNN/ViT models loaded")

            signal_score = float(np.clip(signal_score, 0.0, 1.0))

            return AnalyzerResult(
                analyzer_name=self.name,
                display_name=self.display_name,
                signal_score=signal_score,
                confidence=confidence,
                reasoning=reasoning,
                details={
                    "face_model": FACE_MODEL,
                    "ai_model": AI_MODEL,
                    "face_score": float(face_score),
                    "ai_score": float(ai_score),
                    "method": method,
                },
            )

        except Exception as e:
            logger.warning(f"Dual CNN failed: {e}")
            return AnalyzerResult.failed(self.name, self.display_name, f"Dual CNN failed: {e}")

    def _run_model(self, pil_image, processor, model):
        """Run a single model and return AI probability."""
        import torch

        inputs = processor(images=pil_image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        id2label = model.config.id2label
        ai_prob = 0.0
        for idx, label in id2label.items():
            if any(t in label.lower() for t in ["artificial", "ai", "fake", "generated", "synthetic"]):
                ai_prob = float(probs[idx])
        return ai_prob

    def _load_models(self):
        """Lazy-load both models."""
        from transformers import AutoModelForImageClassification, AutoImageProcessor
        import gc

        gc.collect()
        try:
            from transformers import ViTImageProcessorPil as ImageProc
        except ImportError:
            from transformers import AutoImageProcessor as ImageProc

        logger.info(f"Loading face model: {FACE_MODEL}")
        try:
            self.face_proc = ImageProc.from_pretrained(FACE_MODEL)
            self.face_model = AutoModelForImageClassification.from_pretrained(
                FACE_MODEL, low_cpu_mem_usage=True
            )
            self.face_model.eval()
        except Exception as e:
            logger.warning(f"Failed to load face model {FACE_MODEL}: {e}")
            self.face_model = None

        logger.info(f"Loading AI model: {AI_MODEL}")
        try:
            self.ai_proc = ImageProc.from_pretrained(AI_MODEL)
            self.ai_model = AutoModelForImageClassification.from_pretrained(
                AI_MODEL, low_cpu_mem_usage=True
            )
            self.ai_model.eval()
        except Exception as e:
            logger.warning(f"Failed to load AI model {AI_MODEL}: {e}")
            self.ai_model = None
        gc.collect()
