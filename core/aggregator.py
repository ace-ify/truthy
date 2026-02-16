"""
Aggregator for combining chunk-level predictions into final results.
"""
from typing import List, Dict, Any
from dataclasses import dataclass
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CHUNK_DURATION, AI_THRESHOLD


@dataclass
class ChunkResult:
    chunk_index: int
    start_time: float
    end_time: float
    has_speech: bool
    speech_probability: float
    ai_probability: float | None
    prediction: str | None


@dataclass
class AnalysisResult:
    overall_ai_probability: float
    verdict: str
    confidence: str
    total_chunks: int
    speech_chunks: int
    chunk_results: List[Dict[str, Any]]


def aggregate_results(
    vad_results: List[Dict[str, Any]], 
    detection_results: List[Dict[str, Any]],
    chunk_duration: float = CHUNK_DURATION
) -> AnalysisResult:
    """
    Combine VAD and detection results into final analysis.
    
    Args:
        vad_results: List of VAD results per chunk
        detection_results: List of detection results for speech chunks
        chunk_duration: Duration of each chunk in seconds
        
    Returns:
        AnalysisResult with aggregated predictions
    """
    chunk_results = []
    detection_idx = 0
    
    for i, vad in enumerate(vad_results):
        start_time = i * chunk_duration
        end_time = (i + 1) * chunk_duration
        
        if vad["has_speech"]:
            # Get corresponding detection result
            if detection_idx < len(detection_results):
                det = detection_results[detection_idx]
                ai_prob = det["ai_probability"]
                prediction = det["prediction"]
                detection_idx += 1
            else:
                ai_prob = None
                prediction = None
        else:
            ai_prob = None
            prediction = None
        
        chunk_results.append({
            "chunk_index": i,
            "start_time": round(start_time, 2),
            "end_time": round(end_time, 2),
            "has_speech": vad["has_speech"],
            "speech_probability": round(vad["speech_probability"], 3),
            "ai_probability": round(ai_prob, 3) if ai_prob is not None else None,
            "prediction": prediction
        })
    
    # Calculate overall statistics
    speech_chunks = [r for r in chunk_results if r["has_speech"] and r["ai_probability"] is not None]

    if speech_chunks:
        ai_probs = [r["ai_probability"] for r in speech_chunks]
        
        # Use MAX-based approach: if ANY chunk is strongly AI, flag the whole audio
        max_ai_prob = max(ai_probs)
        avg_ai_prob = sum(ai_probs) / len(ai_probs)
        
        # Count how many chunks are above threshold
        ai_chunk_count = sum(1 for p in ai_probs if p > AI_THRESHOLD)
        ai_chunk_ratio = ai_chunk_count / len(ai_probs)
        
        # Final score logic:
        # 1. If max AI prob > 70%, likely AI (even one strong detection matters)
        # 2. If more than 30% of chunks are AI, likely AI
        # 3. Otherwise use average
        if max_ai_prob > 0.7:
            # Strong AI signal in at least one chunk - weight towards max
            final_ai_prob = max_ai_prob * 0.7 + avg_ai_prob * 0.3
        elif ai_chunk_ratio > 0.3:
            # Multiple chunks flagged as AI
            final_ai_prob = max(avg_ai_prob, max_ai_prob * 0.8)
        else:
            # Use average for borderline cases
            final_ai_prob = avg_ai_prob
    else:
        final_ai_prob = 0.0
    
    # Determine verdict and confidence
    if not speech_chunks:
        verdict = "No Speech Detected"
        confidence = "N/A"
    else:
        verdict = "AI Generated" if final_ai_prob > AI_THRESHOLD else "Human Voice"
        
        # Calculate confidence based on how far from threshold
        distance_from_threshold = abs(final_ai_prob - AI_THRESHOLD)
        if distance_from_threshold > 0.3:
            confidence = "High"
        elif distance_from_threshold > 0.15:
            confidence = "Medium"
        else:
            confidence = "Low"
    
    return AnalysisResult(
        overall_ai_probability=round(final_ai_prob, 3),
        verdict=verdict,
        confidence=confidence,
        total_chunks=len(chunk_results),
        speech_chunks=len(speech_chunks),
        chunk_results=chunk_results
    )
