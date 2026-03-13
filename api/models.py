"""
Pydantic models for API request/response validation.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Literal


class ChunkResultResponse(BaseModel):
    """Response model for individual chunk analysis."""
    chunk_index: int
    start_time: float
    end_time: float
    has_speech: bool
    speech_probability: float
    ai_probability: Optional[float]
    prediction: Optional[str]


class AnalysisResponse(BaseModel):
    """Response model for complete audio analysis."""
    success: bool
    filename: str
    duration_seconds: float
    overall_ai_probability: float
    verdict: str
    confidence: str
    total_chunks: int
    speech_chunks: int
    chunk_results: List[ChunkResultResponse]


class ErrorResponse(BaseModel):
    """Response model for errors."""
    success: bool = False
    error: str
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    models_loaded: bool
    device: str


# ============================================================================
# Models for Base64 API endpoint (per problem statement)
# ============================================================================

class VoiceDetectionRequest(BaseModel):
    """Request model for voice detection API (Base64 input)."""
    language: Literal["Tamil", "English", "Hindi", "Malayalam", "Telugu"] = Field(
        ..., 
        description="Language of the audio: Tamil, English, Hindi, Malayalam, or Telugu"
    )
    audioFormat: Literal["mp3", "wav"] = Field(
        default="mp3",
        description="Audio format (mp3 or wav)"
    )
    audioBase64: str = Field(
        ..., 
        description="Base64-encoded MP3 audio data"
    )


class VoiceDetectionResponse(BaseModel):
    """Success response model for voice detection API."""
    status: Literal["success"] = "success"
    language: str
    classification: Literal["AI_GENERATED", "HUMAN"]
    confidenceScore: float = Field(..., ge=0.0, le=1.0)
    explanation: str


class VoiceDetectionErrorResponse(BaseModel):
    """Error response model for voice detection API."""
    status: Literal["error"] = "error"
    message: str


# ============================================================================
# Image Detection Models
# ============================================================================

class SignalBreakdown(BaseModel):
    """Individual analyzer result in the signal breakdown."""
    analyzer: str
    display_name: str
    score: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    processing_time_ms: int = 0
    error: Optional[str] = None


class ImageDetectionRequest(BaseModel):
    """Request model for image detection (Base64 input)."""
    imageBase64: str = Field(
        ...,
        description="Base64-encoded image data"
    )
    imageFormat: Literal["jpg", "jpeg", "png", "webp", "bmp", "gif"] = Field(
        default="jpg",
        description="Image format"
    )
    mode: Literal["quick", "standard", "thorough"] = Field(
        default="standard",
        description="Detection mode: quick (~2s), standard (~10s), thorough (~25s)"
    )


class ImageDetectionResponse(BaseModel):
    """Response model for image detection."""
    status: Literal["success"] = "success"
    classification: Literal["AI_GENERATED", "HUMAN", "INCONCLUSIVE"]
    confidenceScore: float = Field(..., ge=0.0, le=1.0)
    confidence: str  # "High" | "Medium" | "Low"
    explanation: str
    generatorGuess: Optional[str] = None
    signalBreakdown: List[SignalBreakdown] = []
    heatmapBase64: Optional[str] = None
