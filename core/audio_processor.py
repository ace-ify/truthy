"""
Audio processing utilities for loading, chunking, and preprocessing audio files.
"""
import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr
from pathlib import Path
from typing import List, Tuple
import tempfile
import os

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SAMPLE_RATE, CHUNK_DURATION, ENABLE_NOISE_REDUCTION


def load_audio(file_path: str, target_sr: int = SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    """
    Load an audio file and resample to target sample rate.
    
    Args:
        file_path: Path to the audio file
        target_sr: Target sample rate (default: 16000 Hz for WavLM)
        
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    # Load audio with librosa (handles various formats)
    audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
    return audio, sr


def remove_noise(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Remove background noise from audio using spectral gating.
    
    Args:
        audio: Audio signal as numpy array
        sr: Sample rate
        
    Returns:
        Denoised audio array
    """
    # Apply noise reduction
    # Use a moderate reduction to avoid removing discriminative artifacts
    reduced_noise = nr.reduce_noise(y=audio, sr=sr, prop_decrease=0.4)
    return reduced_noise


def split_into_chunks(audio: np.ndarray, sr: int = SAMPLE_RATE, 
                      chunk_duration: float = CHUNK_DURATION) -> List[np.ndarray]:
    """
    Split audio into fixed-duration chunks.
    
    Args:
        audio: Audio signal as numpy array
        sr: Sample rate
        chunk_duration: Duration of each chunk in seconds
        
    Returns:
        List of audio chunks
    """
    chunk_samples = int(chunk_duration * sr)
    chunks = []
    
    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i:i + chunk_samples]
        # Pad last chunk if necessary
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode='constant')
        chunks.append(chunk)
    
    return chunks


def save_temp_wav(audio: np.ndarray, sr: int = SAMPLE_RATE) -> str:
    """
    Save audio to a temporary WAV file.
    
    Args:
        audio: Audio signal as numpy array
        sr: Sample rate
        
    Returns:
        Path to temporary WAV file
    """
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    sf.write(temp_file.name, audio, sr)
    return temp_file.name


def process_uploaded_file(file_path: str, remove_bg_noise: bool | None = None) -> Tuple[np.ndarray, int]:
    """
    Full preprocessing pipeline for uploaded audio file.
    
    Args:
        file_path: Path to uploaded audio file
        remove_bg_noise: Whether to apply noise reduction
        
    Returns:
        Tuple of (processed_audio, sample_rate)
    """
    # Load and resample
    audio, sr = load_audio(file_path)
    
    # Default to config setting if not explicitly provided
    if remove_bg_noise is None:
        remove_bg_noise = ENABLE_NOISE_REDUCTION

    # Apply noise reduction if requested
    if remove_bg_noise:
        audio = remove_noise(audio, sr)
    
    return audio, sr
