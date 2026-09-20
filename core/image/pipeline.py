"""
Pipeline orchestrator.
Loads analyzers, runs them in parallel, feeds results to the judge.
"""
import logging
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Generator

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import ENABLED_ANALYZERS, ENABLED_MODEL_DETECTORS, ANALYZER_TIMEOUT, JUDGE_MODE

from core.image.preprocessor import ImageData, load_image_from_path, load_image_from_base64
from core.image.analyzers.base import BaseAnalyzer, AnalyzerResult
from core.image.judge import Judge, ImageVerdict, get_judge

logger = logging.getLogger(__name__)


# Registry of available analyzers
ANALYZER_REGISTRY = {
    "frequency": ("core.image.analyzers.frequency", "FrequencyAnalyzer"),
    "ela": ("core.image.analyzers.ela", "ELAAnalyzer"),
    "noise": ("core.image.analyzers.noise", "NoiseAnalyzer"),
    "texture": ("core.image.analyzers.texture", "TextureAnalyzer"),
    "edge": ("core.image.analyzers.edge", "EdgeAnalyzer"),
    "color": ("core.image.analyzers.color", "ColorAnalyzer"),
    "metadata": ("core.image.analyzers.metadata", "MetadataAnalyzer"),
    "compression": ("core.image.analyzers.compression", "CompressionAnalyzer"),
    "statistical": ("core.image.analyzers.statistical", "StatisticalAnalyzer"),
    "geometry": ("core.image.analyzers.geometry", "GeometryAnalyzer"),
    "phase_coherence": ("core.image.analyzers.phase_coherence", "PhaseCoherenceAnalyzer"),
}

MODEL_REGISTRY = {
    "cnn_detector": ("core.image.models.cnn_detector", "CNNDetector"),
    "clip_detector": ("core.image.models.clip_detector", "CLIPDetector"),
    "dire_detector": ("core.image.models.dire", "DIREDetector"),
    "trufor_heatmap": ("core.image.models.trufor", "TruForDetector"),
}


class ImageDetectionPipeline:
    """Runs all enabled analyzers in parallel and produces a final verdict."""

    def __init__(self):
        self.analyzers: List[BaseAnalyzer] = []
        self.judge: Judge = get_judge()
        self._load_analyzers()

    def _load_analyzers(self):
        """Dynamically load enabled analyzers."""
        import importlib

        # Load forensic analyzers
        for name in ENABLED_ANALYZERS:
            if name in ANALYZER_REGISTRY:
                module_path, class_name = ANALYZER_REGISTRY[name]
                try:
                    module = importlib.import_module(module_path)
                    analyzer_class = getattr(module, class_name)
                    self.analyzers.append(analyzer_class())
                    logger.info(f"Loaded analyzer: {name}")
                except Exception as e:
                    logger.warning(f"Failed to load analyzer {name}: {e}")

        # Load model detectors
        for name in ENABLED_MODEL_DETECTORS:
            if name in MODEL_REGISTRY:
                module_path, class_name = MODEL_REGISTRY[name]
                try:
                    module = importlib.import_module(module_path)
                    detector_class = getattr(module, class_name)
                    self.analyzers.append(detector_class())
                    logger.info(f"Loaded model detector: {name}")
                except Exception as e:
                    logger.warning(f"Failed to load model detector {name}: {e}")

        logger.info(f"Pipeline ready with {len(self.analyzers)} analyzers")

    def analyze_image(
        self,
        image_data: ImageData,
        mode: str = "standard",
        judge_mode: Optional[str] = None,
    ) -> ImageVerdict:
        """
        Run the full analysis pipeline.

        Args:
            image_data: Preprocessed image
            mode: "quick" (top 3 fastest), "standard" (all), "thorough" (all + LLM judge)
            judge_mode: Override judge mode ("weighted", "llm", "hybrid")

        Returns:
            ImageVerdict with final result
        """
        start_time = time.perf_counter()

        # Select analyzers based on mode
        if mode == "quick":
            # Quick: run CNN + CLIP (ML models) + frequency + ELA for a fast but ML-backed result
            quick_names = {"cnn_detector", "clip_detector", "frequency", "ela"}
            analyzers_to_run = [a for a in self.analyzers if a.name in quick_names]
            if not analyzers_to_run:
                analyzers_to_run = self.analyzers[:3]  # Fallback if no ML models loaded
        else:
            analyzers_to_run = self.analyzers

        if mode == "thorough":
            judge_mode = judge_mode or JUDGE_MODE  # Respect configured judge mode
        else:
            judge_mode = judge_mode or JUDGE_MODE

        # Run analyzers in parallel
        results = self._run_parallel(analyzers_to_run, image_data)

        # Feed to judge
        verdict = self.judge.evaluate(results, mode=judge_mode)

        total_time = int((time.perf_counter() - start_time) * 1000)
        logger.info(
            f"Pipeline complete in {total_time}ms: {verdict.verdict} "
            f"({verdict.overall_ai_probability:.0%}, {verdict.confidence} confidence)"
        )

        return verdict

    def _run_parallel(
        self, analyzers: List[BaseAnalyzer], image_data: ImageData
    ) -> List[AnalyzerResult]:
        """Run all analyzers in parallel using ThreadPoolExecutor."""
        results: List[AnalyzerResult] = []

        with ThreadPoolExecutor(max_workers=min(len(analyzers), 4)) as executor:
            future_to_analyzer = {
                executor.submit(analyzer.analyze, image_data): analyzer
                for analyzer in analyzers
            }

            try:
                for future in as_completed(future_to_analyzer, timeout=ANALYZER_TIMEOUT + 10):
                    analyzer = future_to_analyzer[future]
                    try:
                        result = future.result(timeout=ANALYZER_TIMEOUT)
                        results.append(result)
                        logger.info(
                            f"  {analyzer.name}: score={result.signal_score:.2f}, "
                            f"conf={result.confidence:.2f}, {result.processing_time_ms}ms"
                        )
                    except Exception as e:
                        logger.warning(f"  {analyzer.name}: FAILED ({e})")
                        results.append(
                            AnalyzerResult.failed(analyzer.name, analyzer.display_name, str(e))
                        )
            except TimeoutError:
                # Some analyzers didn't finish — add failures for them
                for future, analyzer in future_to_analyzer.items():
                    if not future.done():
                        future.cancel()
                        logger.warning(f"  {analyzer.name}: TIMED OUT")
                        results.append(
                            AnalyzerResult.failed(analyzer.name, analyzer.display_name, "Timed out")
                        )

        return results

    def analyze_from_path(self, file_path: str, **kwargs) -> ImageVerdict:
        """Convenience: load image from path and analyze."""
        image_data = load_image_from_path(file_path)
        return self.analyze_image(image_data, **kwargs)

    def analyze_from_base64(self, b64_string: str, image_format: str = "jpeg", **kwargs) -> ImageVerdict:
        """Convenience: load image from base64 and analyze."""
        image_data = load_image_from_base64(b64_string, image_format)
        return self.analyze_image(image_data, **kwargs)

    def stream_analysis(
        self,
        image_data: ImageData,
        mode: str = "standard",
        judge_mode: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """
        Stream analysis results as SSE events.
        Yields JSON strings for each analyzer result as it completes,
        then yields the final verdict.
        """
        start_time = time.perf_counter()

        if mode == "quick":
            quick_names = {"cnn_detector", "clip_detector", "frequency", "ela"}
            analyzers_to_run = [a for a in self.analyzers if a.name in quick_names]
            if not analyzers_to_run:
                analyzers_to_run = self.analyzers[:3]
        else:
            analyzers_to_run = self.analyzers

        if mode == "thorough":
            judge_mode = judge_mode or JUDGE_MODE  # Respect configured judge mode
        else:
            judge_mode = judge_mode or JUDGE_MODE

        results: List[AnalyzerResult] = []
        total = len(analyzers_to_run)

        # Yield start event
        yield json.dumps({
            "type": "start",
            "total_analyzers": total,
            "mode": mode,
        })

        with ThreadPoolExecutor(max_workers=min(total, 4)) as executor:
            future_to_analyzer = {
                executor.submit(analyzer.analyze, image_data): analyzer
                for analyzer in analyzers_to_run
            }

            try:
                for future in as_completed(future_to_analyzer, timeout=ANALYZER_TIMEOUT + 10):
                    analyzer = future_to_analyzer[future]
                    try:
                        result = future.result(timeout=ANALYZER_TIMEOUT)
                        results.append(result)
                        # Yield individual result
                        yield json.dumps({
                            "type": "analyzer_result",
                            "analyzer": result.analyzer_name,
                            "display_name": result.display_name,
                            "score": round(result.signal_score, 3),
                            "confidence": round(result.confidence, 3),
                            "weight": round(result.weight, 2),
                            "reasoning": result.reasoning,
                            "processing_time_ms": result.processing_time_ms,
                            "completed": len(results),
                            "total": total,
                        })
                    except Exception as e:
                        failed = AnalyzerResult.failed(analyzer.name, analyzer.display_name, str(e))
                        results.append(failed)
                        yield json.dumps({
                            "type": "analyzer_result",
                            "analyzer": analyzer.name,
                            "display_name": analyzer.display_name,
                            "score": 0.5,
                            "confidence": 0.0,
                            "weight": 0.0,
                            "reasoning": f"Failed: {e}",
                            "processing_time_ms": 0,
                            "error": str(e),
                            "completed": len(results),
                            "total": total,
                        })
            except TimeoutError:
                for future, analyzer in future_to_analyzer.items():
                    if not future.done():
                        future.cancel()
                        failed = AnalyzerResult.failed(analyzer.name, analyzer.display_name, "Timed out")
                        results.append(failed)

        # Feed to judge and yield final verdict
        verdict = self.judge.evaluate(results, mode=judge_mode)
        total_time = int((time.perf_counter() - start_time) * 1000)

        yield json.dumps({
            "type": "verdict",
            "overall_ai_probability": verdict.overall_ai_probability,
            "verdict": verdict.verdict,
            "confidence": verdict.confidence,
            "explanation": verdict.explanation,
            "generator_guess": verdict.generator_guess,
            "heatmap_b64": verdict.heatmap_b64,
            "total_time_ms": total_time,
        })

    def stream_from_path(self, file_path: str, **kwargs) -> Generator[str, None, None]:
        image_data = load_image_from_path(file_path)
        return self.stream_analysis(image_data, **kwargs)


# Singleton
_pipeline = None

def get_pipeline() -> ImageDetectionPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = ImageDetectionPipeline()
    return _pipeline
