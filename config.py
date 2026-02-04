"""
Configuration settings for AI Voice Detection system.
"""
from pathlib import Path
import os

# Paths
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
MODEL_CACHE_DIR = BASE_DIR / "models"

# Ensure directories exist
UPLOAD_DIR.mkdir(exist_ok=True)
MODEL_CACHE_DIR.mkdir(exist_ok=True)

# Audio Processing Settings
SAMPLE_RATE = 16000  # Required for WavLM
CHUNK_DURATION = 6.0  # seconds per chunk
MIN_SPEECH_PROB = 0.5  # VAD threshold
ENABLE_NOISE_REDUCTION = True  # Enable background noise reduction

# Model Settings
DEEPFAKE_MODEL_ID = "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"  # AI vs Human classifier
VAD_MODEL_REPO = "snakers4/silero-vad"  # Voice Activity Detection

# Detection Thresholds
AI_THRESHOLD = 0.5  # Above this = AI, below = Human

# API Settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
