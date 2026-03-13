"""
Base analyzer interface.
All forensic analyzers and model detectors implement this.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import time

from core.image.preprocessor import ImageData


@dataclass
class AnalyzerResult:
    """Standardized output from any analyzer."""
    analyzer_name: str
    display_name: str
    signal_score: float         # 0.0 = definitely human, 1.0 = definitely AI
    confidence: float           # 0.0-1.0, how sure the analyzer is
    reasoning: str              # Human-readable explanation
    details: dict = field(default_factory=dict)  # Analyzer-specific data
    processing_time_ms: int = 0
    error: Optional[str] = None # Set if analyzer failed
    weight: float = 1.0         # Weight hint for the judge

    @staticmethod
    def failed(name: str, display_name: str, error: str) -> "AnalyzerResult":
        """Create a failed result (excluded from scoring)."""
        return AnalyzerResult(
            analyzer_name=name,
            display_name=display_name,
            signal_score=0.5,  # Neutral
            confidence=0.0,     # Zero confidence = excluded from scoring
            reasoning=f"Analyzer failed: {error}",
            error=error,
        )


class BaseAnalyzer(ABC):
    """Abstract base class for all image analyzers."""

    name: str = "base"
    display_name: str = "Base Analyzer"
    # Weight hint for the judge (higher = more influence)
    weight: float = 1.0

    @abstractmethod
    def _analyze(self, image_data: ImageData) -> AnalyzerResult:
        """
        Run analysis on the image.

        Args:
            image_data: Preprocessed image container

        Returns:
            AnalyzerResult with signal_score, confidence, reasoning, details
        """
        ...

    def analyze(self, image_data: ImageData) -> AnalyzerResult:
        """Run analysis with timing and error handling."""
        start = time.perf_counter()
        try:
            result = self._analyze(image_data)
            result.processing_time_ms = int((time.perf_counter() - start) * 1000)
            result.weight = self.weight
            return result
        except Exception as e:
            elapsed = int((time.perf_counter() - start) * 1000)
            result = AnalyzerResult.failed(self.name, self.display_name, str(e))
            result.processing_time_ms = elapsed
            return result
