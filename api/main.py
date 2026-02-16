"""
FastAPI application for AI Voice Detection.
"""
import os
import base64
import tempfile
import logging
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import SAMPLE_RATE, CHUNK_DURATION, UPLOAD_DIR
from core.audio_processor import process_uploaded_file, split_into_chunks
from core.aggregator import aggregate_results
from api.models import (
    AnalysisResponse, ErrorResponse, HealthResponse,
    VoiceDetectionRequest, VoiceDetectionResponse, VoiceDetectionErrorResponse
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API Key configuration - Read from environment variable
# Set API_KEYS env var in Render (comma-separated for multiple keys)
# Requests from same origin (website) don't need API key
API_KEYS = set(
    key.strip() 
    for key in os.getenv("API_KEYS", "").split(",")
    if key.strip()
)

# Rate limiter configuration
limiter = Limiter(key_func=get_remote_address)

# Create FastAPI app
app = FastAPI(
    title="AI Voice Detector API",
    description="Detect AI-generated voices vs human voices using deep learning",
    version="1.0.0"
)


# Custom exception handler to return proper error format
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Return error responses in the required format: {status: "error", message: "..."}"""
    # If detail is already a dict with status/message, use it directly
    if isinstance(exc.detail, dict) and "status" in exc.detail and "message" in exc.detail:
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.detail
        )
    # Otherwise, wrap the detail as a message
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": str(exc.detail)}
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return validation errors in the required format: {status: "error", message: "..."}"""
    errors = exc.errors()
    if errors:
        # Get the first error and format it nicely
        first_error = errors[0]
        field = ".".join(str(loc) for loc in first_error.get("loc", []) if loc != "body")
        msg = first_error.get("msg", "Validation error")
        message = f"Invalid value for '{field}': {msg}" if field else msg
    else:
        message = "Request validation failed"
    
    return JSONResponse(
        status_code=400,
        content={"status": "error", "message": message}
    )

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiter middleware and handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# Serve static files
STATIC_DIR = Path(__file__).parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Model instances - will be initialized on startup
vad_processor = None
detector = None
models_initialized = False


def init_models():
    global vad_processor, detector, models_initialized
    if not models_initialized:
        logger.info("Loading VAD model...")
        from core.vad import VADProcessor
        vad_processor = VADProcessor()
        
        logger.info("Loading deepfake detector model...")
        from core.detector import DeepfakeDetector
        detector = DeepfakeDetector()
        
        models_initialized = True
        logger.info("All models loaded!")


@app.on_event("startup")
async def startup_event():
    init_models()


def get_models():
    global vad_processor, detector
    if not models_initialized:
        init_models()
    return vad_processor, detector


def verify_api_key(
    request: Request,
    x_api_key: str = Header(None)
) -> str:
    """
    Verify API key from request header.
    Requests from same origin (website) are allowed without API key.
    External API calls require x-api-key header.
    """
    # Check if request is from same origin (our website)
    referer = request.headers.get("referer", "")
    origin = request.headers.get("origin", "")
    host = request.headers.get("host", "")
    
    # Allow requests from same origin without API key
    if host and (host in referer or host in origin):
        return "same-origin"
    
    # External requests require API key
    if x_api_key is None:
        raise HTTPException(
            status_code=401,
            detail={"status": "error", "message": "Missing API key. Provide x-api-key header."}
        )
    if x_api_key not in API_KEYS:
        raise HTTPException(
            status_code=401,
            detail={"status": "error", "message": "Invalid API key"}
        )
    return x_api_key


def generate_explanation(ai_prob: float, classification: str, language: str) -> str:
    if classification == "AI_GENERATED":
        if ai_prob > 0.9:
            return f"Strong synthetic voice patterns detected. Audio shows clear signs of AI generation with unnatural pitch consistency and robotic speech artifacts in {language}."
        elif ai_prob > 0.75:
            return f"High probability of AI-generated speech. Detected irregular prosody and mechanical intonation patterns typical of synthetic voices in {language}."
        else:
            return f"Moderate indicators of AI generation detected. Some unnatural speech patterns and timing irregularities found in the {language} audio sample."
    else:
        if ai_prob < 0.2:
            return f"Audio exhibits natural human speech characteristics. Normal breathing patterns, natural pitch variations, and organic voice modulation detected in {language}."
        elif ai_prob < 0.35:
            return f"Strong indicators of authentic human voice. Natural prosody and speech patterns consistent with genuine {language} speech."
        else:
            return f"Audio appears to be human speech with natural voice characteristics typical of {language} speakers."


@app.get("/")
async def root():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    try:
        vad, det = get_models()
        return HealthResponse(
            status="healthy",
            models_loaded=True,
            device=str(det.device)
        )
    except Exception as e:
        return HealthResponse(
            status="degraded",
            models_loaded=False,
            device="unknown"
        )


@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_audio(
    file: UploadFile = File(...),
    remove_noise: bool = True
):
    """
    Analyze an uploaded audio file for AI-generated voice detection.
    
    Args:
        file: Audio file (MP3, WAV, M4A, etc.)
        remove_noise: Whether to apply noise reduction preprocessing
        
    Returns:
        Analysis results with AI probability and per-chunk breakdown
    """
    # Validate file type
    allowed_extensions = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".wma", ".aac"}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Allowed: {allowed_extensions}"
        )
    
    # Save uploaded file temporarily
    temp_path = None
    try:
        # Create temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        # Load models
        vad, det = get_models()
        
        # Process audio
        logger.info(f"Processing file: {file.filename}")
        audio, sr = process_uploaded_file(temp_path, remove_bg_noise=remove_noise)
        duration = len(audio) / sr
        
        # Split into chunks
        chunks = split_into_chunks(audio, sr, CHUNK_DURATION)
        logger.info(f"Split into {len(chunks)} chunks of {CHUNK_DURATION}s each")
        
        # Run VAD on each chunk
        vad_results = vad.process_chunks(chunks, sr)
        speech_chunk_count = sum(1 for r in vad_results if r["has_speech"])
        logger.info(f"Speech detected in {speech_chunk_count}/{len(chunks)} chunks")
        
        # Run detector only on chunks with speech
        detection_results = []
        for i, (chunk, vad_result) in enumerate(zip(chunks, vad_results)):
            if vad_result["has_speech"]:
                ai_prob, prediction = det.predict(chunk, sr)
                detection_results.append({
                    "chunk_index": i,
                    "ai_probability": ai_prob,
                    "prediction": prediction
                })
                logger.info(f"  Chunk {i}: {prediction} ({ai_prob:.1%})")
        
        # Aggregate results
        result = aggregate_results(vad_results, detection_results, CHUNK_DURATION)
        
        logger.info(f"Verdict: {result.verdict} (AI prob: {result.overall_ai_probability:.1%})")
        
        return AnalysisResponse(
            success=True,
            filename=file.filename,
            duration_seconds=round(duration, 2),
            overall_ai_probability=result.overall_ai_probability,
            verdict=result.verdict,
            confidence=result.confidence,
            total_chunks=result.total_chunks,
            speech_chunks=result.speech_chunks,
            chunk_results=result.chunk_results
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing audio: {str(e)}"
        )
    
    finally:
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


# ============================================================================
# Voice Detection API (Base64 Input - Per Problem Statement)
# ============================================================================

@limiter.limit("100/minute")
@app.post("/api/voice-detection", response_model=VoiceDetectionResponse)
async def voice_detection(
    request: VoiceDetectionRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Detect whether a voice sample is AI-generated or Human.
    
    This endpoint accepts Base64-encoded MP3 audio and returns the classification
    result with confidence score and explanation.
    
    Supports: Tamil, English, Hindi, Malayalam, Telugu
    
    Requires API key in x-api-key header.
    """
    temp_path = None
    try:
        # Decode Base64 audio
        try:
            audio_bytes = base64.b64decode(request.audioBase64)
        except Exception:
            raise HTTPException(
                status_code=400,
                detail={"status": "error", "message": "Invalid Base64 encoding for audio data"}
            )
        
        # Validate minimum audio size
        if len(audio_bytes) < 1000:
            raise HTTPException(
                status_code=400,
                detail={"status": "error", "message": "Audio data too small or corrupt"}
            )
        
        # Save to temp file
        file_ext = f".{request.audioFormat}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_file.write(audio_bytes)
            temp_path = temp_file.name
        
        # Load models
        vad, det = get_models()
        
        # Process audio
        logger.info(f"Processing Base64 audio for {request.language}")
        audio, sr = process_uploaded_file(temp_path, remove_bg_noise=True)
        duration = len(audio) / sr
        logger.info(f"Audio duration: {duration:.2f}s")
        
        # Split into chunks
        chunks = split_into_chunks(audio, sr, CHUNK_DURATION)
        logger.info(f"Split into {len(chunks)} chunks")
        
        # Run VAD on each chunk
        vad_results = vad.process_chunks(chunks, sr)
        speech_chunk_count = sum(1 for r in vad_results if r["has_speech"])
        logger.info(f"Speech detected in {speech_chunk_count}/{len(chunks)} chunks")
        
        if speech_chunk_count == 0:
            raise HTTPException(
                status_code=400,
                detail={"status": "error", "message": "No speech detected in audio sample"}
            )
        
        # Run detector on chunks with speech
        detection_results = []
        for i, (chunk, vad_result) in enumerate(zip(chunks, vad_results)):
            if vad_result["has_speech"]:
                ai_prob, prediction = det.predict(chunk, sr)
                detection_results.append({
                    "chunk_index": i,
                    "ai_probability": ai_prob,
                    "prediction": prediction
                })
        
        # Aggregate results
        result = aggregate_results(vad_results, detection_results, CHUNK_DURATION)
        
        # Determine classification
        classification = "AI_GENERATED" if result.overall_ai_probability > 0.5 else "HUMAN"
        confidence_score = round(result.overall_ai_probability if classification == "AI_GENERATED" 
                                  else 1 - result.overall_ai_probability, 2)
        
        # Generate explanation
        explanation = generate_explanation(
            result.overall_ai_probability, 
            classification, 
            request.language
        )
        
        logger.info(f"Result: {classification} (confidence: {confidence_score})")
        
        return VoiceDetectionResponse(
            status="success",
            language=request.language,
            classification=classification,
            confidenceScore=confidence_score,
            explanation=explanation
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error processing audio: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"status": "error", "message": f"Error processing audio: {str(e)}"}
        )
    
    finally:
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


@app.get("/api/info")
async def api_info():
    return {
        "name": "Truthy AI Voice Detector API",
        "version": "1.0.0",
        "supported_languages": ["Tamil", "English", "Hindi", "Malayalam", "Telugu"],
        "endpoints": {
            "POST /api/voice-detection": "Analyze Base64 audio (requires API key)",
            "POST /api/analyze": "Analyze uploaded audio file (legacy)",
            "GET /api/health": "Check API health and model status"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
