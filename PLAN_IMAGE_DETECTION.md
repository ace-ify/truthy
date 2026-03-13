# Truthy Image Detection - Implementation Plan

## Architecture Overview

Extend the existing Truthy architecture with an image detection module that mirrors the audio detection pattern: **Analyzers (parallel) -> Aggregator/Judge -> Verdict**.

```
truthy/
├── api/
│   ├── main.py                    # Add image detection routes
│   └── models.py                  # Add image request/response schemas
├── core/
│   ├── detector.py                # [existing] Audio detector
│   ├── vad.py                     # [existing] Voice Activity Detection
│   ├── audio_processor.py         # [existing] Audio preprocessing
│   ├── aggregator.py              # [existing] Audio result aggregation
│   │
│   └── image/                     # [NEW] Image detection package
│       ├── __init__.py
│       ├── pipeline.py            # Orchestrator - runs all analyzers, feeds judge
│       ├── judge.py               # LLM/VLM judge - aggregates signals into verdict
│       ├── preprocessor.py        # Image loading, normalization, format handling
│       │
│       ├── analyzers/             # Individual forensic analyzers
│       │   ├── __init__.py
│       │   ├── base.py            # BaseAnalyzer ABC
│       │   ├── frequency.py       # FFT/spectral analysis
│       │   ├── ela.py             # Error Level Analysis
│       │   ├── noise.py           # Noise pattern / PRNU analysis
│       │   ├── texture.py         # Texture & LBP analysis
│       │   ├── edge.py            # Edge/gradient analysis
│       │   ├── color.py           # Color channel correlation
│       │   ├── metadata.py        # EXIF/metadata forensics
│       │   ├── compression.py     # JPEG artifact analysis
│       │   ├── statistical.py     # Benford's law, histogram, chi-square
│       │   └── geometry.py        # Shadow, reflection, perspective checks
│       │
│       └── models/                # ML model-based detectors
│           ├── __init__.py
│           ├── base.py            # BaseModelDetector ABC
│           ├── clip_detector.py   # CLIP-based UnivFD detection
│           ├── trufor.py          # TruFor forensics model
│           ├── dire.py            # Diffusion Reconstruction Error
│           └── cnn_detector.py    # CNN binary classifier (custom trained)
│
├── config.py                      # Add image detection settings
└── static/                        # Add image detection UI page
```

---

## Phase 1: Foundation (Core Infrastructure)

### 1.1 Config Updates (`config.py`)

Add image-specific settings:
- `IMAGE_MAX_SIZE = 4096` (max dimension)
- `IMAGE_UPLOAD_MAX_MB = 20`
- `IMAGE_AI_THRESHOLD = 0.5`
- `JUDGE_MODEL = "gemini-2.0-flash"` (or Claude via API)
- `HF_TOKEN` reuse from existing config
- `ANALYZERS_ENABLED` list to toggle analyzers
- `ANALYZER_TIMEOUT = 30` seconds per analyzer

### 1.2 Image Preprocessor (`core/image/preprocessor.py`)

- Load image from file path or base64
- Normalize to RGB, handle RGBA/grayscale/CMYK
- Resize if over max dimension (preserve aspect ratio)
- Convert to numpy array and PIL Image (both needed by different analyzers)
- Extract raw bytes for compression analysis

### 1.3 Base Analyzer Interface (`core/image/analyzers/base.py`)

```python
class BaseAnalyzer(ABC):
    name: str           # "frequency_analysis"
    display_name: str   # "Frequency Analysis"

    @abstractmethod
    def analyze(self, image: np.ndarray, pil_image: Image, metadata: dict) -> AnalyzerResult:
        """Returns signal_score (0-1), confidence (0-1), reasoning (str), details (dict)"""
```

All analyzers return the same `AnalyzerResult` dataclass:
```python
@dataclass
class AnalyzerResult:
    analyzer_name: str
    signal_score: float      # 0.0 = definitely human, 1.0 = definitely AI
    confidence: float        # How confident this analyzer is in its result
    reasoning: str           # Human-readable explanation
    details: dict            # Analyzer-specific data (frequencies, heatmaps, etc.)
    processing_time_ms: int
```

### 1.4 Pipeline Orchestrator (`core/image/pipeline.py`)

- Loads enabled analyzers from config
- Runs all analyzers in parallel using `asyncio.gather` or `concurrent.futures.ThreadPoolExecutor`
- Collects all `AnalyzerResult`s
- Passes results to the Judge
- Returns final `ImageAnalysisResult`
- Has timeout per analyzer (skip if too slow)

---

## Phase 2: Forensic Analyzers (Traditional Signal Processing)

Build these in order of detection value:

### 2.1 Frequency/FFT Analyzer (`frequency.py`)
- Compute 2D FFT of the image
- Analyze high-frequency spectral patterns
- GAN images show characteristic periodic peaks in Fourier domain
- Compute spectral energy distribution and compare to natural image statistics
- Look for radial symmetry artifacts (GAN fingerprint)

### 2.2 Error Level Analysis (`ela.py`)
- Re-save image as JPEG at a fixed quality (e.g., 95%)
- Compute pixel-wise difference between original and resaved
- AI images show different ELA patterns than real photos
- Calculate uniformity score of error levels across regions
- Segment into grid and compare regional consistency

### 2.3 Noise Pattern Analyzer (`noise.py`)
- Extract noise residual using wavelet denoising
- Analyze noise pattern uniformity (real cameras have non-uniform sensor noise)
- Check for PRNU-like patterns (camera fingerprint)
- AI images have either too-uniform or synthetic noise patterns
- Compute noise variance across image blocks

### 2.4 Texture Analyzer (`texture.py`)
- Compute Local Binary Patterns (LBP) histogram
- Analyze micro-texture statistics
- Compare to known distributions of natural vs AI textures
- Check for repeating texture patterns (common in AI)

### 2.5 Edge/Gradient Analyzer (`edge.py`)
- Compute Sobel/Canny edges
- Analyze gradient direction consistency
- Check for unnatural edge sharpness (AI tends to have different edge profiles)
- Laplacian variance analysis

### 2.6 Color Channel Analyzer (`color.py`)
- Analyze RGB channel correlations
- Check color histogram distributions
- Compare chrominance patterns to natural image statistics
- Detect unusual color quantization

### 2.7 Metadata Analyzer (`metadata.py`)
- Extract EXIF data using Pillow/piexif
- Check for camera info (make, model, lens, GPS)
- Detect AI generator markers or suspicious absence of metadata
- Check for C2PA content credentials
- Low weight in final score (easily faked)

### 2.8 JPEG Compression Analyzer (`compression.py`)
- Detect JPEG quality level
- Analyze DCT coefficient distributions
- Check for double compression artifacts
- Detect grid alignment inconsistencies

### 2.9 Statistical Analyzer (`statistical.py`)
- Benford's law analysis on pixel values
- Chi-square test on LSBs
- Histogram shape analysis
- Compare to natural image statistics models

### 2.10 Geometry Analyzer (`geometry.py`)
- Shadow direction consistency (using edge detection + gradient analysis)
- Perspective line convergence check
- Reflection physics plausibility
- Primarily useful for photos with clear geometric cues

---

## Phase 3: ML Model Detectors

### 3.1 CLIP-based Detector (`clip_detector.py`)
- Use CLIP ViT features as input to a binary classifier
- Based on UnivFD (Universal Fake Detector) approach
- Works across multiple generators without retraining
- Use HuggingFace model: `openai/clip-vit-large-patch14` + linear probe
- Or use pre-trained UnivFD weights if available

### 3.2 TruFor Integration (`trufor.py`)
- Integrate the TruFor model (academic paper by Guillaro et al.)
- Provides pixel-level forensic heatmap + integrity score
- Can detect both AI generation and manipulation
- Use HuggingFace or direct model weights
- Returns heatmap as base64 for visualization

### 3.3 DIRE - Diffusion Reconstruction Error (`dire.py`)
- Pass image through a diffusion model's encoder -> decoder
- Compare reconstruction to original (MSE/SSIM)
- AI images reconstruct near-perfectly, real photos don't
- This exploits a fundamental property of diffusion models
- Use a lightweight VAE (Stable Diffusion's VAE) for the reconstruction
- Can use HuggingFace Inference API to avoid local GPU

### 3.4 CNN Binary Classifier (`cnn_detector.py`)
- Fine-tuned ResNet/EfficientNet on AI vs real dataset
- Trained on diverse generators (DALL-E, Midjourney, SD, Flux, etc.)
- Can use HuggingFace pre-trained models
- Candidates: `umm-maybe/AI-image-detector`, similar HF models
- Same API pattern as existing audio detector (HF API + local fallback)

---

## Phase 4: The Judge (`core/image/judge.py`)

### LLM-as-Judge Architecture

The Judge receives all analyzer results and produces the final verdict.

**Input to Judge:**
```python
{
    "analyzers": [
        {
            "name": "frequency_analysis",
            "signal_score": 0.82,
            "confidence": 0.75,
            "reasoning": "High-frequency spectral peaks detected consistent with GAN generation patterns"
        },
        {
            "name": "noise_analysis",
            "signal_score": 0.15,
            "confidence": 0.60,
            "reasoning": "Noise patterns appear natural, consistent with camera sensor noise"
        },
        # ... all other analyzers
    ],
    "metadata_summary": "No EXIF data found, JPEG quality 92",
    "image_properties": {"width": 1024, "height": 1024, "format": "JPEG"}
}
```

**Judge Implementation (3 modes, configurable):**

1. **Weighted Ensemble (fast, no API needed)**
   - Weighted average of analyzer scores based on confidence
   - Conflict detection: if analyzers disagree strongly, reduce overall confidence
   - Override rules: if any high-confidence analyzer gives extreme score, it can override
   - Similar to existing `aggregator.py` pattern

2. **LLM Judge (accurate, needs API)**
   - Send structured prompt to Gemini Flash / Claude with all analyzer results
   - Ask it to reason about conflicts, weigh evidence, and produce final verdict
   - Structured output: verdict, probability, confidence, explanation, generator_guess
   - Benefits: can reason about semantic contradictions, explain in natural language

3. **Hybrid (recommended default)**
   - First compute weighted ensemble score
   - If score is borderline (0.35-0.65), escalate to LLM judge for reasoning
   - If score is confident (>0.85 or <0.15), use weighted ensemble directly
   - Best tradeoff of speed, cost, and accuracy

**Judge Output:**
```python
@dataclass
class ImageVerdict:
    overall_ai_probability: float  # 0.0-1.0
    verdict: str                   # "AI Generated" | "Human Created" | "Inconclusive"
    confidence: str                # "High" | "Medium" | "Low"
    explanation: str               # Natural language explanation
    generator_guess: str | None    # "Likely Midjourney" | "Likely Stable Diffusion" | None
    signal_breakdown: list         # Per-analyzer results for transparency
    heatmap_b64: str | None        # TruFor heatmap if available
```

---

## Phase 5: API & Frontend

### 5.1 API Endpoints

Add to `api/main.py`:

**`POST /api/image-detection`** (Base64 input, mirrors voice-detection)
```json
Request:
{
    "imageBase64": "...",
    "imageFormat": "jpg",
    "mode": "standard"        // "quick" | "standard" | "thorough"
}

Response:
{
    "status": "success",
    "classification": "AI_GENERATED",
    "confidenceScore": 0.87,
    "explanation": "Multiple forensic signals indicate AI generation...",
    "generatorGuess": "Likely Midjourney v6",
    "signalBreakdown": [
        {"analyzer": "frequency_analysis", "score": 0.82, "reasoning": "..."},
        {"analyzer": "noise_analysis", "score": 0.15, "reasoning": "..."}
    ],
    "heatmapBase64": "..."
}
```

**`POST /api/image-analyze`** (File upload)
- Same as above but accepts multipart file upload

**Detection Modes:**
- `quick` — Only top-3 fastest analyzers + weighted ensemble (< 2s)
- `standard` — All forensic analyzers + ML models + hybrid judge (5-15s)
- `thorough` — Everything + LLM judge always + multiple model passes (15-30s)

### 5.2 Pydantic Models (`api/models.py`)

Add `ImageDetectionRequest`, `ImageDetectionResponse`, `SignalBreakdown` models.

### 5.3 Frontend

Add a new page or tab for image detection in `static/`:
- Drag & drop image upload
- Show progressive results (analyzer by analyzer)
- Display signal breakdown with visual indicators
- Show TruFor heatmap overlay on the image
- Confidence gauge/meter
- Expandable explanation section

---

## Phase 6: Implementation Order (Priority)

Build in this exact order for fastest path to working product:

| Step | What | Why | Est. Effort |
|------|------|-----|-------------|
| 1 | `preprocessor.py` + `base.py` + `pipeline.py` | Foundation everything plugs into | Small |
| 2 | `frequency.py` + `ela.py` + `noise.py` | 3 highest-value forensic analyzers | Medium |
| 3 | `cnn_detector.py` (HF API) | Instant ML detection via API | Small |
| 4 | Weighted ensemble judge | Working end-to-end pipeline | Small |
| 5 | API endpoints + basic frontend | Usable product | Medium |
| 6 | `texture.py` + `edge.py` + `color.py` | More forensic signals | Medium |
| 7 | `metadata.py` + `compression.py` + `statistical.py` | Supporting signals | Medium |
| 8 | `clip_detector.py` (UnivFD) | Strong cross-generator detection | Medium |
| 9 | `dire.py` | Most powerful theoretical approach | Hard |
| 10 | `trufor.py` | Heatmap visualization | Hard |
| 11 | LLM judge integration | Reasoning + explanation quality | Medium |
| 12 | Progressive SSE streaming | Real-time result updates | Medium |

Steps 1-5 give us a working MVP. Steps 6-12 make it competitive/superior.

---

## Dependencies to Add

```
# Image processing
Pillow>=10.0.0
opencv-python-headless>=4.8.0

# Forensic analysis
scikit-image>=0.21.0

# ML models (reuse existing torch/transformers)
# torch (already installed)
# transformers (already installed)

# LLM Judge
google-generativeai>=0.8.0   # Gemini Flash API (free tier available)
# OR anthropic>=0.30.0       # Claude API alternative

# EXIF extraction
piexif>=1.1.3
```
