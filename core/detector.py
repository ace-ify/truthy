"""
Deepfake audio detector using WavLM-based model.
Single model approach for maximum accuracy and simplicity.
"""
import torch
import numpy as np
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from typing import Tuple
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SAMPLE_RATE, DEEPFAKE_MODEL_ID, AI_THRESHOLD


class DeepfakeDetector:
    """AI Voice Detection using fine-tuned WavLM model."""
    
    def __init__(self, model_id: str = DEEPFAKE_MODEL_ID):
        self.model_id = model_id
        self.model = None
        self.feature_extractor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ai_index = None
        self.human_index = None
        self._load_model()
    
    def _load_model(self):
        """Load the deepfake detection model with memory optimizations."""
        import gc
        print(f"Loading model: {self.model_id}...")
        print(f"Using device: {self.device} (CPU forced for memory efficiency)")
        
        # Explicitly clear any existing memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        try:
            # Load feature extractor
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                self.model_id,
                trust_remote_code=True
            )
            
            # Load model with memory-efficient flags
            print("📦 Compressing model weights (Dynamic Quantization)...")
            self.model = AutoModelForAudioClassification.from_pretrained(
                self.model_id,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            # Apply Dynamic Quantization (Int8)
            # This is key to fitting 1.2GB model into 512MB RAM
            self.model = torch.quantization.quantize_dynamic(
                self.model, 
                {torch.nn.Linear}, 
                dtype=torch.qint8
            )
            
            # Ensure model is on CPU (Quantization is CPU-optimized)
            self.model.to("cpu")
            self.model.eval()
            
            # Final cleanup
            gc.collect()
            
            # Resolve label indices
            id2label = self.model.config.id2label
            for idx, label in id2label.items():
                label_lower = label.lower()
                if any(term in label_lower for term in ["fake", "spoof", "deepfake", "synthetic", "ai"]):
                    self.ai_index = idx
                if any(term in label_lower for term in ["real", "bonafide", "human", "genuine"]):
                    self.human_index = idx
                    
        except Exception as e:
            print(f"Error loading detector model: {e}")
            raise e
            
        print("✅ Deepfake detector loaded successfully.")
        
        # Fallback if labels not found (generic LABEL_0, LABEL_1)
        # For ASVSpoof5 models: 0 = spoof (AI), 1 = bonafide (human)
        if self.ai_index is None:
            self.ai_index = 0  # ASVSpoof convention: 0 = spoof/fake
        if self.human_index is None:
            self.human_index = 1  # ASVSpoof convention: 1 = bonafide/real
            
        print(f"Label mapping: AI={self.ai_index}, Human={self.human_index}")
        print("Model loaded successfully!")
    
    def predict(self, audio: np.ndarray, sr: int = SAMPLE_RATE) -> Tuple[float, str]:
        """
        Predict whether audio is AI-generated or human.
        
        Args:
            audio: Audio signal as numpy array
            sr: Sample rate
            
        Returns:
            Tuple of (ai_probability, label)
            - ai_probability: 0.0 (definitely human) to 1.0 (definitely AI)
            - label: "AI" or "Human"
        """
        # Extract features
        inputs = self.feature_extractor(
            audio, 
            sampling_rate=sr, 
            return_tensors="pt",
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
        
        # Get AI probability
        probs = probabilities[0].cpu().numpy()
        ai_prob = float(probs[self.ai_index])
        
        # Determine label
        label = "AI" if ai_prob > AI_THRESHOLD else "Human"
        
        return ai_prob, label
    
    def get_model_labels(self) -> dict:
        """Return the model's label mapping."""
        return self.model.config.id2label


# Singleton instance for reuse
_detector = None

def get_detector() -> DeepfakeDetector:
    """Get or create singleton detector instance."""
    global _detector
    if _detector is None:
        _detector = DeepfakeDetector()
    return _detector
