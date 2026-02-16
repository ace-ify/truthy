"""
Deepfake audio detector using WavLM-based model.
Primary: HuggingFace Inference API (no local memory needed)
Fallback: Local model loading with quantization

IMPORTANT: This module uses LAZY IMPORTS to minimize memory usage
when running in API-only mode (no torch needed for cloud inference).
"""
import numpy as np
import requests
import os
import io
from typing import Tuple, Optional
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SAMPLE_RATE, DEEPFAKE_MODEL_ID, AI_THRESHOLD


class DeepfakeDetector:
    
    # HuggingFace Inference API endpoint
    HF_API_URL = f"https://api-inference.huggingface.co/models/{DEEPFAKE_MODEL_ID}"
    
    def __init__(self, model_id: str = DEEPFAKE_MODEL_ID):
        self.model_id = model_id
        self.model = None
        self.feature_extractor = None
        self.device = None  # Set later only if using local mode
        self.ai_index = None
        self.human_index = None
        self.use_api = False  # Will be set during initialization
        self.hf_token = os.environ.get("HF_TOKEN", "")
        
        self._initialize()
    
    def _initialize(self):
        """Initialize detector - try API first, fall back to local."""
        print(f"Initializing detector with {self.model_id}")
        
        # Check if HF token is available
        if self.hf_token:
            print("HF token found, testing API...")
            if self._test_api():
                self.use_api = True
                print("Using HuggingFace Inference API")
                # Default label mapping for API mode
                self.ai_index = 0
                self.human_index = 1
                return
            else:
                print("API test failed, falling back to local model")
        else:
            print("No HF_TOKEN set, loading model locally")
        
        # Fallback to local model (this imports torch)
        self._load_model_local()
    
    def _test_api(self) -> bool:
        try:
            headers = {"Authorization": f"Bearer {self.hf_token}"}
            # Send a minimal test request with timeout
            response = requests.post(
                self.HF_API_URL,
                headers=headers,
                json={"inputs": "test"},
                timeout=15
            )
            print(f"   API test response: {response.status_code}")
            # API returns 200 or 503 (model loading) - both are acceptable
            if response.status_code in [200, 503]:
                return True
            print(f"   Unexpected API status: {response.status_code} - {response.text[:200]}")
            return False
        except Exception as e:
            print(f"   API test error: {e}")
            return False
    
    def _load_model_local(self):
        """Load the deepfake detection model locally with memory optimizations.
        
        NOTE: This imports torch and transformers only when needed (lazy import).
        """
        # LAZY IMPORTS - only load heavy libraries if we need local mode
        print("Loading PyTorch and Transformers...")
        import torch
        from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
        import gc
        
        self.device = torch.device("cpu")
        print(f"Loading model: {self.model_id}")
        print(f"   Using device: {self.device}")
        
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
            print("Applying dynamic quantization...")
            self.model = AutoModelForAudioClassification.from_pretrained(
                self.model_id,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            # Apply Dynamic Quantization (Int8)
            self.model = torch.quantization.quantize_dynamic(
                self.model, 
                {torch.nn.Linear}, 
                dtype=torch.qint8
            )
            
            self.model.to("cpu")
            self.model.eval()
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
            
        # Fallback label mapping
        if self.ai_index is None:
            self.ai_index = 0
        if self.human_index is None:
            self.human_index = 1
            
        print(f"Model loaded. Labels: AI={self.ai_index}, Human={self.human_index}")
    
    def _predict_api(self, audio: np.ndarray, sr: int) -> Tuple[float, str]:
        import soundfile as sf
        
        # Convert audio to WAV bytes
        buffer = io.BytesIO()
        sf.write(buffer, audio, sr, format='WAV')
        buffer.seek(0)
        audio_bytes = buffer.read()
        
        headers = {"Authorization": f"Bearer {self.hf_token}"}
        
        # Retry logic for model loading (503 status)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.HF_API_URL,
                    headers=headers,
                    data=audio_bytes,
                    timeout=120
                )
                
                if response.status_code == 200:
                    results = response.json()
                    # Parse API response
                    # Format: [{"label": "LABEL_0", "score": 0.95}, {"label": "LABEL_1", "score": 0.05}]
                    ai_prob = 0.0
                    for item in results:
                        label = item.get("label", "").lower()
                        score = item.get("score", 0.0)
                        # Check if this is the AI/fake label
                        if any(term in label for term in ["fake", "spoof", "deepfake", "synthetic", "ai", "label_0", "0"]):
                            ai_prob = score
                            break
                    
                    label = "AI" if ai_prob > AI_THRESHOLD else "Human"
                    return ai_prob, label
                    
                elif response.status_code == 503:
                    # Model is loading, wait and retry
                    print(f"   Model loading on HF servers (attempt {attempt + 1}/{max_retries})")
                    import time
                    time.sleep(20)  # Wait for model to load
                else:
                    print(f"   API error: {response.status_code} - {response.text[:200]}")
                    break
            except requests.exceptions.Timeout:
                print(f"   Timed out (attempt {attempt + 1}/{max_retries})")
                continue
            except Exception as e:
                print(f"   API exception: {e}")
                break
        
        raise Exception(f"HuggingFace API failed after {max_retries} attempts")
    
    def _predict_local(self, audio: np.ndarray, sr: int) -> Tuple[float, str]:
        import torch
        
        inputs = self.feature_extractor(
            audio, 
            sampling_rate=sr, 
            return_tensors="pt",
            padding=True
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
        
        probs = probabilities[0].cpu().numpy()
        ai_prob = float(probs[self.ai_index])
        label = "AI" if ai_prob > AI_THRESHOLD else "Human"
        
        return ai_prob, label
    
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
        if self.use_api:
            return self._predict_api(audio, sr)
        else:
            return self._predict_local(audio, sr)
    
    def get_model_labels(self) -> dict:
        if self.model:
            return self.model.config.id2label
        return {0: "LABEL_0 (AI/Fake)", 1: "LABEL_1 (Human/Real)"}


# Singleton instance for reuse
_detector = None

def get_detector() -> DeepfakeDetector:
    global _detector
    if _detector is None:
        _detector = DeepfakeDetector()
    return _detector
