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
from fastapi.responses import FileResponse, JSONResponse, Response, HTMLResponse
from fastapi.exceptions import RequestValidationError
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import SAMPLE_RATE, CHUNK_DURATION, UPLOAD_DIR, PREWARM_IMAGE_MODELS, REVIEWER_KEY
from core.audio_processor import process_uploaded_file, split_into_chunks
from core.aggregator import aggregate_results
from core.report import envelope_from_image, envelope_from_video, envelope_from_voice
from core.report_html import render_report_html
from core.governance import store as case_store
from core.governance import bundle as case_bundle
from api.models import (
    AnalysisResponse, ErrorResponse, HealthResponse,
    VoiceDetectionRequest, VoiceDetectionResponse, VoiceDetectionErrorResponse,
    ImageDetectionRequest, ImageDetectionResponse, SignalBreakdown,
    VideoDetectionRequest, VideoDetectionResponse,
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
    """Fast non-blocking startup: serve web UI immediately, lazy-load models on demand."""
    logger.info("Truthy engine initialized. Models will lazy-load on first analysis request.")
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
    """Plain, honest, pay-context explanation. We report the model's probability and a
    safe next action — we do NOT fabricate specific acoustic 'artifacts' we didn't measure."""
    if classification == "AI_GENERATED":
        return (
            f"This {language} voice sample is likely AI-generated or cloned "
            f"({ai_prob:.0%} from the voice model). Do not act on any payment or account "
            f"instructions in it — confirm with the person on a known, trusted number first."
        )
    return (
        f"This {language} voice sample is likely a real human voice "
        f"({1 - ai_prob:.0%} from the voice model). No strong signs of AI cloning were found, "
        f"but for high-value or urgent money requests, still verify through a trusted channel."
    )


@app.get("/")
@app.get("/audio")
async def root():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/image")
async def image_page():
    return FileResponse(str(STATIC_DIR / "image.html"))


@app.get("/video")
async def video_page():
    return FileResponse(str(STATIC_DIR / "video.html"))


@app.get("/verify")
async def verify_page():
    return FileResponse(str(STATIC_DIR / "verify.html"))


@app.get("/review")
async def review_page():
    return FileResponse(str(STATIC_DIR / "review.html"))


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    favicon_path = STATIC_DIR / "favicon.ico"
    if favicon_path.exists():
        return FileResponse(str(favicon_path))
    return Response(status_code=204)


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
            "POST /api/video-detection": "Detect deepfake video (Base64, requires API key)",
            "POST /api/video-analyze": "Detect deepfake video (file upload)",
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


# ============================================================================
# Video Detection API (Biological rPPG + Kinematics)
# ============================================================================

@limiter.limit("30/minute")
@app.post("/api/video-detection", response_model=VideoDetectionResponse)
async def video_detection(
    request: VideoDetectionRequest,
    api_key: str = Depends(verify_api_key),
):
    """
    Detect deepfake video manipulation using optical remote photoplethysmography (rPPG)
    and audio-visual phoneme-viseme biomechanical kinematics (Base64 input).
    """
    temp_path = None
    try:
        try:
            video_bytes = base64.b64decode(request.videoBase64)
        except Exception:
            raise HTTPException(
                status_code=400,
                detail={"status": "error", "message": "Invalid Base64 encoding for video data"}
            )

        if len(video_bytes) < 1000:
            raise HTTPException(
                status_code=400,
                detail={"status": "error", "message": "Video data too small or corrupt"}
            )

        max_bytes = 50 * 1024 * 1024  # 50MB limit
        if len(video_bytes) > max_bytes:
            raise HTTPException(
                status_code=400,
                detail={"status": "error", "message": f"Video exceeds max size of {max_bytes // (1024*1024)}MB"}
            )

        file_ext = f".{request.videoFormat}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_file.write(video_bytes)
            temp_path = temp_file.name

        from core.video.pipeline import get_video_pipeline
        vpipe = get_video_pipeline()
        verdict = vpipe.analyze_video(temp_path)

        return VideoDetectionResponse(
            status="success",
            classification=verdict.verdict,
            confidenceScore=round(
                verdict.overall_ai_probability if verdict.verdict == "AI_GENERATED"
                else 1.0 - verdict.overall_ai_probability if verdict.verdict == "HUMAN"
                else 0.5,
                2
            ),
            confidence=verdict.confidence,
            explanation=verdict.explanation,
            fps=verdict.fps,
            durationSeconds=verdict.duration_seconds,
            totalFramesAnalyzed=verdict.total_frames,
            hasBiologicalPulse=verdict.rppg_results.get("has_pulse", False),
            detectedBpm=verdict.rppg_results.get("bpm"),
            pulseSnr=verdict.rppg_results.get("pulse_snr", 0.0),
            biomechanicalViolations=verdict.lip_sync_results.get("biomechanical_violations", 0),
            bvpChartBase64=verdict.bvp_chart_b64,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error processing video: {e}")
        raise HTTPException(
            status_code=500,
            detail={"status": "error", "message": f"Error processing video: {str(e)}"}
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


@app.post("/api/video-analyze", response_model=VideoDetectionResponse)
async def video_analyze(
    file: UploadFile = File(...),
):
    """
    Detect deepfake video manipulation using optical remote photoplethysmography (rPPG)
    and audio-visual phoneme-viseme biomechanical kinematics (File upload).
    """
    allowed_extensions = {".mp4", ".webm", ".avi", ".mov", ".mkv"}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail={"status": "error", "message": f"Unsupported format: {file_ext}. Allowed: {allowed_extensions}"}
        )

    temp_path = None
    try:
        content = await file.read()
        max_bytes = 50 * 1024 * 1024
        if len(content) > max_bytes:
            raise HTTPException(
                status_code=400,
                detail={"status": "error", "message": f"Video exceeds max size of {max_bytes // (1024*1024)}MB"}
            )

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name

        from core.video.pipeline import get_video_pipeline
        vpipe = get_video_pipeline()
        verdict = vpipe.analyze_video(temp_path)

        return VideoDetectionResponse(
            status="success",
            classification=verdict.verdict,
            confidenceScore=round(
                verdict.overall_ai_probability if verdict.verdict == "AI_GENERATED"
                else 1.0 - verdict.overall_ai_probability if verdict.verdict == "HUMAN"
                else 0.5,
                2
            ),
            confidence=verdict.confidence,
            explanation=verdict.explanation,
            fps=verdict.fps,
            durationSeconds=verdict.duration_seconds,
            totalFramesAnalyzed=verdict.total_frames,
            hasBiologicalPulse=verdict.rppg_results.get("has_pulse", False),
            detectedBpm=verdict.rppg_results.get("bpm"),
            pulseSnr=verdict.rppg_results.get("pulse_snr", 0.0),
            biomechanicalViolations=verdict.lip_sync_results.get("biomechanical_violations", 0),
            bvpChartBase64=verdict.bvp_chart_b64,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error processing video: {e}")
        raise HTTPException(
            status_code=500,
            detail={"status": "error", "message": f"Error processing video: {str(e)}"}
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


from fastapi import Form


def verify_reviewer_key(x_reviewer_key: str = Header(None)) -> str:
    """Gate the human-review queue and case mutations. Fails CLOSED: if no REVIEWER_KEY is
    configured, these endpoints are disabled. Does NOT accept the spoofable same-origin bypass."""
    if not REVIEWER_KEY:
        raise HTTPException(
            status_code=403,
            detail={"status": "error", "message": "Review queue is disabled (set REVIEWER_KEY)."},
        )
    if x_reviewer_key != REVIEWER_KEY:
        raise HTTPException(
            status_code=401,
            detail={"status": "error", "message": "Missing or invalid x-reviewer-key."},
        )
    return "reviewer"


_AUDIO_EXT = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".wma", ".aac"}
_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
_VIDEO_EXT = {".mp4", ".webm", ".avi", ".mov", ".mkv"}


def _detect_media_type(ext: str) -> str:
    if ext in _AUDIO_EXT:
        return "voice"
    if ext in _IMAGE_EXT:
        return "image"
    if ext in _VIDEO_EXT:
        return "video"
    return "unknown"


@app.post("/api/verify")
async def verify(
    request: Request,
    file: UploadFile = File(...),
    consent: str = Form("false"),
    language: str = Form("English"),
    mode: str = Form("standard"),
):
    """Unified 'verify before you pay' endpoint: accepts a voice note, video, or image,
    returns ONE report envelope, and records an auditable case (verdict + SHA-256, not media)."""
    # Consent gate (defense in depth beyond the UI checkbox).
    if str(consent).lower() not in ("true", "1", "yes", "on"):
        raise HTTPException(
            status_code=400,
            detail={"status": "error", "message": "Consent required: confirm you are authorized to analyze this media."},
        )

    ext = Path(file.filename or "").suffix.lower()
    media_type = _detect_media_type(ext)
    if media_type == "unknown":
        raise HTTPException(
            status_code=400,
            detail={"status": "error", "message": f"Unsupported file type: {ext or 'unknown'}."},
        )

    content = await file.read()
    if len(content) < 100:
        raise HTTPException(status_code=400, detail={"status": "error", "message": "File too small or corrupt."})
    max_bytes = (20 if media_type == "image" else 50) * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=400,
            detail={"status": "error", "message": f"File too large. Max {max_bytes // (1024*1024)}MB for {media_type}."},
        )

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tf:
            tf.write(content)
            temp_path = tf.name

        if media_type == "image":
            global image_pipeline
            if image_pipeline is None:
                from core.image.pipeline import get_pipeline
                image_pipeline = get_pipeline()
            verdict = image_pipeline.analyze_from_path(temp_path, mode=mode)
            envelope = envelope_from_image(verdict, filename=file.filename)

        elif media_type == "video":
            from core.video.pipeline import get_video_pipeline
            verdict = get_video_pipeline().analyze_video(temp_path)
            envelope = envelope_from_video(verdict, filename=file.filename)

        else:  # voice
            vad, det = get_models()
            audio, sr = process_uploaded_file(temp_path, remove_bg_noise=True)
            chunks = split_into_chunks(audio, sr, CHUNK_DURATION)
            vad_results = vad.process_chunks(chunks, sr)
            detection_results = []
            for i, (chunk, vr) in enumerate(zip(chunks, vad_results)):
                if vr["has_speech"]:
                    ai_prob, pred = det.predict(chunk, sr)
                    detection_results.append({"chunk_index": i, "ai_probability": ai_prob, "prediction": pred})
            if not detection_results:
                # No speech -> don't guess.
                from core.report import build_envelope
                envelope = build_envelope(
                    media_type="voice", verdict="INCONCLUSIVE", confidence="Low",
                    ai_probability=0.5, explanation="No clear speech was found in this audio, so it can't be verified. Confirm the request through a trusted channel.",
                    model_backed=False, escalate=True, filename=file.filename,
                )
            else:
                result = aggregate_results(vad_results, detection_results, CHUNK_DURATION)
                classification = "AI_GENERATED" if result.overall_ai_probability > 0.5 else "HUMAN"
                explanation = generate_explanation(result.overall_ai_probability, classification, language)
                envelope = envelope_from_voice(result, language, explanation, filename=file.filename)

        # Persist an auditable case (metadata + SHA-256, never the media itself).
        meta = case_store.record_case(envelope, content)
        envelope.update(meta)  # caseId, sha256, createdAt, reviewStatus
        envelope["status"] = "success"
        return JSONResponse(content=envelope)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Verify failed: {e}")
        raise HTTPException(status_code=500, detail={"status": "error", "message": f"Analysis failed: {str(e)}"})
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


@app.get("/api/case/{case_id}")
async def get_case_endpoint(case_id: str):
    case = case_store.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail={"status": "error", "message": "Case not found"})
    import json as _json
    env = _json.loads(case.get("report_json") or "{}")
    env.update({
        "caseId": case["case_id"], "sha256": case["sha256"], "createdAt": case["created_at"],
        "reviewStatus": case["review_status"], "reviewerDecision": case.get("reviewer_decision"),
        "reviewerNote": case.get("reviewer_note"), "status": "success",
    })
    return JSONResponse(content=env)


@app.get("/api/case/{case_id}/report", response_class=HTMLResponse)
async def case_report_html(case_id: str):
    case = case_store.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail={"status": "error", "message": "Case not found"})
    return HTMLResponse(content=render_report_html(case))


@app.post("/api/case/{case_id}/escalate")
async def escalate_case_endpoint(case_id: str):
    if not case_store.get_case(case_id):
        raise HTTPException(status_code=404, detail={"status": "error", "message": "Case not found"})
    case_store.escalate_case(case_id)
    return {"status": "success", "caseId": case_id, "reviewStatus": "queued"}


@app.get("/api/case/{case_id}/bundle")
async def case_bundle_endpoint(case_id: str):
    """Download the forwardable, tamper-evident evidence bundle (JSON) for a case."""
    case = case_store.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail={"status": "error", "message": "Case not found"})
    bundle = case_bundle.build_bundle(case)
    return JSONResponse(content=bundle, headers={
        "Content-Disposition": f'attachment; filename="truthy-{case_id}.bundle.json"'
    })


@app.post("/api/case/{case_id}/report")
async def report_case_endpoint(case_id: str):
    """Forward a case to the fraud/response channel (the Track-3 reporting path).
    Builds a signed bundle, POSTs it to FRAUD_WEBHOOK_URL if configured, and records
    the case as reported either way. Open to the end user (reporting a scam is the point)."""
    case = case_store.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail={"status": "error", "message": "Case not found"})
    bundle = case_bundle.build_bundle(case)
    delivered, ref = case_bundle.forward_bundle(bundle)
    case_store.mark_reported(case_id, ref)
    return {
        "status": "success", "caseId": case_id, "reportRef": ref,
        "delivered": delivered, "signed": bundle["integrity"]["signed"],
        "message": ("Forwarded to the fraud-response channel." if delivered
                    else "Recorded as reported. Download the bundle to forward it, or set FRAUD_WEBHOOK_URL for automatic delivery."),
    }


@app.get("/api/review")
async def review_queue(reviewer: str = Depends(verify_reviewer_key)):
    return {"status": "success", "cases": case_store.list_cases(review_status="queued")}


@app.post("/api/case/{case_id}/review")
async def review_case_endpoint(
    case_id: str,
    decision: str = Form(...),
    note: str = Form(""),
    reviewer: str = Depends(verify_reviewer_key),
):
    if decision not in ("confirm", "override_authentic", "override_fake"):
        raise HTTPException(status_code=400, detail={"status": "error", "message": "Invalid decision"})
    if not case_store.review_case(case_id, decision, note):
        raise HTTPException(status_code=404, detail={"status": "error", "message": "Case not found"})
    return {"status": "success", "caseId": case_id, "reviewStatus": "reviewed", "decision": decision}


if __name__ == "__main__":
    import uvicorn
    from config import API_HOST, API_PORT
    uvicorn.run(app, host=API_HOST, port=API_PORT)
