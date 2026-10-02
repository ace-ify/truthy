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
# The real dual-CNN detector is hardcoded in core/image/models/cnn_detector.py:
#   FACE_MODEL = "dima806/deepfake_vs_real_image_detection"  (face-specific)
#   AI_MODEL   = "Ateeqq/ai-vs-human-image-detector"         (diffusion-aware)
# (The former IMAGE_CNN_MODEL_ID="Organika/sdxl-detector" was never used — removed.)
CLIP_MODEL_ID = "openai/clip-vit-large-patch14"  # only referenced by the disabled clip_detector

# Calibration: Platt-scaling params fit offline over the labeled validation set.
# Missing file => Judge falls back to an identity map and labels output "uncalibrated".
CALIBRATION_PATH = os.getenv("CALIBRATION_PATH", str(BASE_DIR / "calibration.json"))

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
    "phase_coherence",
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

# ============================================================================
# Governance / Evidence Settings (Track 3)
# ============================================================================
# SQLite case ledger: stores verdict metadata + SHA-256 of the analyzed media,
# never the media itself (transient-processing privacy posture).
CASES_DB_PATH = os.getenv("CASES_DB_PATH", str(BASE_DIR / "cases.db"))

# Shared secret required to view the human-review queue and mutate cases
# (approve / override / escalate). Do NOT rely on the same-origin bypass for these.
# Empty => review endpoints are disabled (fail closed).
REVIEWER_KEY = os.getenv("REVIEWER_KEY", "").strip()

# Optional HMAC key to sign exported case bundles — makes forwarded evidence tamper-evident
# so a receiving bank / fraud team can verify integrity. Empty => bundle is unsigned (disclosed).
REPORT_SIGNING_KEY = os.getenv("REPORT_SIGNING_KEY", "").strip()

# Optional webhook to forward a reported case to (e.g. a bank fraud-ops channel).
# Empty => the case is recorded as reported locally without an outbound call.
FRAUD_WEBHOOK_URL = os.getenv("FRAUD_WEBHOOK_URL", "").strip()
