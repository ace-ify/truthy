"""
Voice Activity Detection (VAD) module using Silero VAD.
"""
import torch
import numpy as np
from typing import List, Dict, Any
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SAMPLE_RATE, MIN_SPEECH_PROB


class VADProcessor:
    
    # Silero VAD requires specific window sizes
    WINDOW_SIZE_SAMPLES = 512  # For 16kHz
    
    def __init__(self):
        self.model = None
        self._load_model()
    
    def _load_model(self):
        print("Loading VAD model...")
        
        try:
            # Method 1: Standard Torch Hub load
            model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False,
                trust_repo=True
            )
            self.model = model
            print("VAD model loaded via Torch Hub")
        except Exception as e:
            print(f"Torch Hub load failed: {e}, trying direct download...")
            
            # Method 2: Direct JIT fallback
            try:
                # Use the direct raw link that we know works
                url = "https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.jit"
                model_path = Path("silero_vad.jit")
                
                if not model_path.exists():
                    print(f"Downloading VAD model from {url}")
                    torch.hub.download_url_to_file(url, str(model_path))
                
                # Load as a scripted model
                self.model = torch.jit.load(str(model_path))
                print("VAD model loaded (JIT fallback)")
            except Exception as fe:
                print(f"VAD fallback also failed: {fe}")
                raise fe
                
        if self.model:
            self.model.eval()
    
    def detect_speech_in_chunk(self, audio: np.ndarray, sr: int = SAMPLE_RATE) -> float:
        """
        Detect speech probability in audio chunk by processing small windows.
        
        Args:
            audio: Audio signal as numpy array
            sr: Sample rate
            
        Returns:
            Speech probability (0.0 to 1.0) based on speech coverage
        """
        if sr not in [8000, 16000]:
            raise ValueError(f"Sample rate must be 8000 or 16000, got {sr}")
        
        # Window size depends on sample rate
        window_size = 256 if sr == 8000 else 512
        
        # Reset model state for new audio
        self.model.reset_states()
        
        # Process audio in small windows
        speech_probs = []
        
        for i in range(0, len(audio), window_size):
            chunk_window = audio[i:i + window_size]
            
            # Skip if window is too small
            if len(chunk_window) < window_size:
                break
            
            # Convert to tensor
            audio_tensor = torch.from_numpy(chunk_window).float()
            
            # Get speech probability for this window
            try:
                with torch.no_grad():
                    prob = self.model(audio_tensor, sr).item()
                speech_probs.append(prob)
            except Exception as e:
                print(f"VAD window error: {e}")
                continue
        
        if not speech_probs:
            return 0.5  # Default to uncertain
        
        # Calculate overall speech probability
        # Use max-based approach: if any window has high speech prob, consider it speech
        max_prob = max(speech_probs)
        avg_prob = sum(speech_probs) / len(speech_probs)
        
        # Weight towards max (if there's clear speech anywhere, count it)
        return 0.7 * max_prob + 0.3 * avg_prob
    
    def has_speech(self, audio: np.ndarray, sr: int = SAMPLE_RATE, 
                   threshold: float = MIN_SPEECH_PROB) -> bool:
        prob = self.detect_speech_in_chunk(audio, sr)
        return prob > threshold
    
    def process_chunks(self, chunks: List[np.ndarray], sr: int = SAMPLE_RATE) -> List[Dict[str, Any]]:
        results = []
        for i, chunk in enumerate(chunks):
            # Reset model between chunks for clean state
            self.model.reset_states()
            
            speech_prob = self.detect_speech_in_chunk(chunk, sr)
            results.append({
                "chunk_index": i,
                "speech_probability": speech_prob,
                "has_speech": speech_prob > MIN_SPEECH_PROB
            })
        return results


# Singleton instance for reuse
_vad_processor = None

def get_vad_processor() -> VADProcessor:
    global _vad_processor
    if _vad_processor is None:
        _vad_processor = VADProcessor()
    return _vad_processor
