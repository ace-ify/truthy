"""
Configuration settings for AI Detection system (Voice + Image).
"""
import logging
from pathlib import Path
import os
from dotenv import load_dotenv

# Load .env file from project root (works regardless of working directory)
_env_path = Path(__file__).parent / ".env"
load_dotenv(_env_path, override=True)

# Suppress noisy third-party loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

# Paths
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
MODEL_CACHE_DIR = BASE_DIR / "models"

# Ensure directories exist
UPLOAD_DIR.mkdir(exist_ok=True)
MODEL_CACHE_DIR.mkdir(exist_ok=True)

# ============================================================================
# Audio Detection Settings
# ============================================================================
SAMPLE_RATE = 16000  # Required for WavLM
CHUNK_DURATION = 6.0  # seconds per chunk
MIN_SPEECH_PROB = 0.5  # VAD threshold
ENABLE_NOISE_REDUCTION = True  # Enable background noise reduction

# Audio Model Settings
DEEPFAKE_MODEL_ID = "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"
VAD_MODEL_REPO = "snakers4/silero-vad"

# Audio Detection Thresholds
AI_THRESHOLD = 0.5  # Above this = AI, below = Human

# ============================================================================
# Image Detection Settings
# ============================================================================
IMAGE_MAX_DIMENSION = 4096      # Max width/height before downscaling
IMAGE_UPLOAD_MAX_MB = 20        # Max upload size in MB
IMAGE_AI_THRESHOLD = 0.5        # Above this = AI, below = Human
ANALYZER_TIMEOUT = 30           # Seconds before an analyzer is killed
ELA_QUALITY = 90                # JPEG quality for Error Level Analysis resave
NOISE_BLOCK_SIZE = 64           # Block size for noise pattern analysis

# Image Model Settings
IMAGE_CNN_MODEL_ID = "Organika/sdxl-detector"    # Swin-based, trained on modern AI outputs
CLIP_MODEL_ID = "openai/clip-vit-large-patch14"      # CLIP for UnivFD-style detection

# Judge Settings — "hybrid" uses weighted ensemble + LLM fallback for borderline cases
JUDGE_MODE = os.getenv("JUDGE_MODE", "weighted")  # "weighted" | "llm" | "hybrid"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
JUDGE_LLM_MODEL = "gemini-2.0-flash"

# Groq Settings (free alternative LLM judge)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()

# Which LLM provider to use for the judge: "gemini" or "groq"
# Auto-detected: uses Groq if GROQ_API_KEY is set, else Gemini if GEMINI_API_KEY is set
JUDGE_LLM_PROVIDER = os.getenv("JUDGE_LLM_PROVIDER", "auto")

# Which analyzers to run (toggle on/off)
ENABLED_ANALYZERS = [
    "frequency",
    "ela",
    "noise",
    "texture",
    "edge",
    "color",
    "metadata",
    "compression",
    "statistical",
    "geometry",
]

ENABLED_MODEL_DETECTORS = [
    "cnn_detector",
    # "clip_detector",   # Disabled: zero-shot CLIP has no discriminative power (always says AI ~97%)
    # "dire_detector",   # Disabled: DCT approximation always gives 0.825, no discrimination
    "trufor_heatmap",
]

# ============================================================================
# API Settings
# ============================================================================
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "9000"))

# Pre-warm image models on startup (downloads ~2GB on first run, fast after cached)
# Set to "true" in production so the first request isn't slow
PREWARM_IMAGE_MODELS = os.getenv("PREWARM_IMAGE_MODELS", "false").lower() == "true"
