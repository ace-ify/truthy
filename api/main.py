"""
FastAPI application for AI Voice Detection.
"""
import os
import base64
import tempfile
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.exceptions import RequestValidationError
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import SAMPLE_RATE, CHUNK_DURATION, UPLOAD_DIR, PREWARM_IMAGE_MODELS
from core.audio_processor import process_uploaded_file, split_into_chunks
from core.aggregator import aggregate_results
from api.models import (
    AnalysisResponse, ErrorResponse, HealthResponse,
    VoiceDetectionRequest, VoiceDetectionResponse, VoiceDetectionErrorResponse,
    ImageDetectionRequest, ImageDetectionResponse, SignalBreakdown,
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


@asynccontextmanager
async def lifespan(app):
    """Load all models on startup."""
    init_models()
    yield


# Create FastAPI app
app = FastAPI(
    title="Truthy AI Detection API",
    description="Detect AI-generated voices and images using deep learning ensemble",
    version="2.0.0",
    lifespan=lifespan,
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
image_pipeline = None
models_initialized = False


def init_models():
    global vad_processor, detector, image_pipeline, models_initialized
    if not models_initialized:
        logger.info("Loading VAD model...")
        from core.vad import VADProcessor
        vad_processor = VADProcessor()

        logger.info("Loading deepfake detector model...")
        from core.detector import DeepfakeDetector
        detector = DeepfakeDetector()

        logger.info("Loading image detection pipeline...")
        from core.image.pipeline import ImageDetectionPipeline
        image_pipeline = ImageDetectionPipeline()

        # Pre-warm image ML models so first request isn't slow
        if PREWARM_IMAGE_MODELS:
            logger.info("Pre-warming image ML models (CNN + CLIP)...")
            for analyzer in image_pipeline.analyzers:
                if hasattr(analyzer, '_load_local_model') and analyzer.model is None:
                    try:
                        analyzer._load_local_model()
                        logger.info(f"  Pre-warmed: {analyzer.name}")
                    except Exception as e:
                        logger.warning(f"  Failed to pre-warm {analyzer.name}: {e}")

        models_initialized = True
        logger.info("All models loaded!")






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


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    favicon_path = STATIC_DIR / "favicon.ico"
    if favicon_path.exists():
        return FileResponse(str(favicon_path))
    return Response(status_code=204)


@app.get("/image")
async def image_page():
    return FileResponse(str(STATIC_DIR / "image.html"))


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    try:
        vad, det = get_models()
        return HealthResponse(
            status="healthy",
            models_loaded=True,
            device=str(det.device) if det.device else "api"
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
        "name": "Truthy AI Detection API",
        "version": "2.0.0",
        "supported_languages": ["Tamil", "English", "Hindi", "Malayalam", "Telugu"],
        "endpoints": {
            "POST /api/voice-detection": "Analyze Base64 audio (requires API key)",
            "POST /api/analyze": "Analyze uploaded audio file (legacy)",
            "POST /api/image-detection": "Detect AI-generated images (Base64, requires API key)",
            "POST /api/image-analyze": "Detect AI-generated images (file upload)",
            "POST /api/image-stream": "Stream image analysis results as SSE (file upload)",
            "GET /api/health": "Check API health and model status"
        }
    }


# ============================================================================
# Image Detection API
# ============================================================================

@limiter.limit("60/minute")
@app.post("/api/image-detection", response_model=ImageDetectionResponse)
async def image_detection(
    request: ImageDetectionRequest,
    api_key: str = Depends(verify_api_key),
):
    """
    Detect whether an image is AI-generated or human-created.

    Uses a multi-signal forensic ensemble: frequency analysis, error level analysis,
    noise patterns, CNN classifier, and more — aggregated by a weighted judge.

    Modes:
    - quick: ~2s, top 3 fastest analyzers
    - standard: ~10s, all analyzers
    - thorough: ~25s, all analyzers + LLM reasoning
    """
    try:
        # Validate base64
        try:
            import base64 as b64module
            image_bytes = b64module.b64decode(request.imageBase64)
        except Exception:
            raise HTTPException(
                status_code=400,
                detail={"status": "error", "message": "Invalid Base64 encoding for image data"}
            )

        if len(image_bytes) < 100:
            raise HTTPException(
                status_code=400,
                detail={"status": "error", "message": "Image data too small or corrupt"}
            )

        max_bytes = 20 * 1024 * 1024  # 20MB
        if len(image_bytes) > max_bytes:
            raise HTTPException(
                status_code=400,
                detail={"status": "error", "message": f"Image too large. Max {max_bytes // (1024*1024)}MB"}
            )

        # Run pipeline
        global image_pipeline
        if image_pipeline is None:
            from core.image.pipeline import get_pipeline
            image_pipeline = get_pipeline()

        verdict = image_pipeline.analyze_from_base64(
            request.imageBase64,
            image_format=request.imageFormat,
            mode=request.mode,
        )

        # Map verdict to classification enum
        if verdict.verdict == "AI Generated":
            classification = "AI_GENERATED"
        elif verdict.verdict == "Human Created":
            classification = "HUMAN"
        else:
            classification = "INCONCLUSIVE"

        confidence_score = round(
            verdict.overall_ai_probability if classification == "AI_GENERATED"
            else 1 - verdict.overall_ai_probability if classification == "HUMAN"
            else 0.5,
            2,
        )

        return ImageDetectionResponse(
            status="success",
            classification=classification,
            confidenceScore=confidence_score,
            confidence=verdict.confidence,
            explanation=verdict.explanation,
            generatorGuess=verdict.generator_guess,
            signalBreakdown=[
                SignalBreakdown(**s) for s in verdict.signal_breakdown
            ],
            heatmapBase64=verdict.heatmap_b64,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error processing image: {e}")
        raise HTTPException(
            status_code=500,
            detail={"status": "error", "message": f"Error processing image: {str(e)}"}
        )


@app.post("/api/image-analyze", response_model=ImageDetectionResponse)
async def image_analyze(
    file: UploadFile = File(...),
    mode: str = "standard",
):
    """
    Analyze an uploaded image file for AI generation detection.
    File upload version (no API key needed for same-origin).
    """
    allowed_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail={"status": "error", "message": f"Unsupported format: {file_ext}. Allowed: {allowed_extensions}"}
        )

    temp_path = None
    try:
        content = await file.read()

        max_bytes = 20 * 1024 * 1024
        if len(content) > max_bytes:
            raise HTTPException(
                status_code=400,
                detail={"status": "error", "message": f"Image too large. Max {max_bytes // (1024*1024)}MB"}
            )

        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name

        # Run pipeline
        global image_pipeline
        if image_pipeline is None:
            from core.image.pipeline import get_pipeline
            image_pipeline = get_pipeline()

        verdict = image_pipeline.analyze_from_path(temp_path, mode=mode)

        if verdict.verdict == "AI Generated":
            classification = "AI_GENERATED"
        elif verdict.verdict == "Human Created":
            classification = "HUMAN"
        else:
            classification = "INCONCLUSIVE"

        confidence_score = round(
            verdict.overall_ai_probability if classification == "AI_GENERATED"
            else 1 - verdict.overall_ai_probability if classification == "HUMAN"
            else 0.5,
            2,
        )

        return ImageDetectionResponse(
            status="success",
            classification=classification,
            confidenceScore=confidence_score,
            confidence=verdict.confidence,
            explanation=verdict.explanation,
            generatorGuess=verdict.generator_guess,
            signalBreakdown=[
                SignalBreakdown(**s) for s in verdict.signal_breakdown
            ],
            heatmapBase64=verdict.heatmap_b64,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error processing image: {e}")
        raise HTTPException(
            status_code=500,
            detail={"status": "error", "message": f"Error processing image: {str(e)}"}
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


# ============================================================================
# SSE Streaming Endpoint
# ============================================================================

@app.post("/api/image-stream")
async def image_stream(
    file: UploadFile = File(...),
    mode: str = "standard",
):
    """
    Stream image analysis results as Server-Sent Events.
    Each analyzer result is sent as it completes, followed by the final verdict.
    """
    from fastapi.responses import StreamingResponse
    from core.image.preprocessor import load_image_from_path as _load_path

    allowed_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail={"status": "error", "message": f"Unsupported format: {file_ext}"}
        )

    content = await file.read()
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail={"status": "error", "message": "Image too large. Max 20MB"}
        )

    # Save to temp file
    temp_path = None
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
        temp_file.write(content)
        temp_path = temp_file.name

    def event_generator():
        try:
            global image_pipeline
            if image_pipeline is None:
                from core.image.pipeline import get_pipeline
                image_pipeline = get_pipeline()

            image_data = _load_path(temp_path)
            for event_data in image_pipeline.stream_analysis(image_data, mode=mode):
                yield f"data: {event_data}\n\n"
            yield "data: {\"type\": \"done\"}\n\n"
        except Exception as e:
            import json
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn
    from config import API_HOST, API_PORT
    uvicorn.run(app, host=API_HOST, port=API_PORT)
