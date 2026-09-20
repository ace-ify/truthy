<p align="center">
  <img src="https://see.fontimg.com/api/rf5/rv0aB/ZTRlNjgyYThiNTRjNDE1ZWFiYzliZmY5OGI5MDhhM2Yub3Rm/dHJ1dGh5/teknaf-regular.png?r=fs&h=81&w=1250&fg=3b82f6&bg=000000&s=65" alt="Truthy Logo" height="60">
</p>

# Truthy — High-Performance Multimodal Deepfake & Synthetic Media Forensic System (Voice · Vision · Video)

[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8.svg?logo=opencv&logoColor=white)](https://opencv.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E.svg?logo=huggingface&logoColor=black)](https://huggingface.co)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Truthy** is an enterprise-grade multimodal synthetic media detection engine. Built to counter the proliferation of generative deepfakes across voice, vision, and video, Truthy abandons brittle, monolithic black-box classifiers in favor of a **first-principles, multi-signal hybrid forensic architecture**: marrying deep acoustic/visual representations with physical and biological realities:
* **Video Forensics:** Biological hemodynamics via **Remote Photoplethysmography (rPPG)** pulse extraction and **Phoneme-Viseme** speech kinematics.
* **Vision Forensics:** **Bi-spectral Fourier phase coherence**, dual-CNN vision classifiers (ViT + Swin), 11 digital signal processing (DSP) analyzers, and spatial manipulation heatmaps.
* **Audio Forensics:** **Glottal inverse filtering (LPC)** for vocal cord aerodynamic flow, Silero VAD speech gating, and quantized cross-lingual Wav2Vec2-XLSR acoustic representations.

---

## 🏛️ System Architecture

```
                                  ┌────────────────────────┐
                                  │   MULTIMODAL INGEST    │
                                  │   (REST API / SSE)     │
                                  └───────────┬────────────┘
                                              │
                     ┌────────────────────────┼────────────────────────┐
                     ▼                        ▼                        ▼
        ┌─────────────────────────┐ ┌─────────────────────────┐ ┌─────────────────────────┐
        │     AUDIO PIPELINE      │ │     VISION PIPELINE     │ │     VIDEO PIPELINE      │
        └────────────┬────────────┘ └────────────┬────────────┘ └────────────┬────────────┘
                     │                                   │                                   │
        ┌────────────▼────────────┐         ┌────────────▼────────────┐         ┌────────────▼────────────┐
        │  Spectral De-Noising    │         │ Image Preprocessing &   │         │ Sequential Frame Ingest │
        │  & 16kHz Resampling     │         │ Spatial Normalization   │         │ & Audio Track Splitting │
        └────────────┬────────────┘         └────────────┬────────────┘         └────────────┬────────────┘
                     │                                   │                                   │
        ┌────────────▼────────────┐         ┌────────────▼────────────┐         ┌────────────┼────────────┐
        │   Silero VAD (PyTorch)  │         │   ThreadPoolExecutor    │         │            │            │
        │ Active Speech Isolation │         │   Parallel Signal Pool  │         ▼            ▼            ▼
        └────────────┬────────────┘         └────────────┬────────────┘   ┌───────────┐┌───────────┐┌───────────┐
                     │                                   │                │Optical BVP││ Phoneme-  ││Keyframe   │
        ┌────────────▼────────────┐         ┌────────────┼────────────┐   │Pulse rPPG ││ Viseme    ││Forensic   │
        │  LPC Glottal Inversion  │         ▼            ▼            ▼   │(Hemoglobin││Kinematic  ││Spatial    │
        │  & GCI Phase Jitter     │   ┌───────────┐┌───────────┐┌────────┐│Capillary) ││Synchrony  ││Analyzers  │
        └────────────┬────────────┘   │ Dual-CNN  ││ 11x DSP   ││TruFor  │└─────┬─────┘└─────┬─────┘└─────┬─────┘
                     │                │ Vision    ││ Forensics ││Heatmap │      └────────────┼────────────┘
        ┌────────────▼────────────┐   │(ViT+Swin) ││ (FFT/ELA/ ││(Spatial│                   │
        │  Wav2Vec2-XLSR (300M)   │   │           ││ Phase-Coh)││Anomaly)│                   │
        │ Acoustic Representation │   └─────┬─────┘└─────┬─────┘└────┬───┘                   │
        │ (INT8 Dynamic Quantized)│         └────────────┼───────────┘                       │
        └────────────┬────────────┘                      │                                   │
                     │                                   ▼                                   │
        ┌────────────▼────────────┐         ┌─────────────────────────┐                      │
        │   Non-linear Quadratic  │         │   Bayesian Consensus    │                      │
        │ Chunk Splicing Weighting│         │  + LLM Judge Arbiter    │                      │
        └────────────┬────────────┘         │  (Gemini / Groq Llama)  │                      │
                     │                      └────────────┬────────────┘                      │
                     ▼                                   ▼                                   ▼
        ┌───────────────────────────────────────────────────────────────────────────────────────────┐
        │                       AUDITABLE VERDICT, REASONING & PROOFS                               │
        │  • Tri-State Verdict: AI_GENERATED / HUMAN / INCONCLUSIVE                                 │
        │  • Calibrated Confidence Score (0.00 – 1.00)                                              │
        │  • Biological Proof: Optical Blood Volume Pulse (BVP) Waveform & Estimated BPM            │
        │  • Kinematic Proof: Plosive Acoustic-to-Viseme Aperture Discrepancy Timeline              │
        │  • Spatial Proof: RGB Manipulation Heatmap & Bi-Spectral Phase Coupling Diagnostics       │
        └───────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Core Methodologies

### 1. Acoustic Deepfake Engine
* **Acoustic Representation:** Powered by `wav2vec2-large-xlsr-deepfake-audio-classification`, fine-tuned across multilingual synthetic and vocoded speech benchmarks.
* **Dual Execution Modes:**
  * *Cloud Inference:* Zero-RAM serverless offloading via Hugging Face Inference API.
  * *Edge/Container Inference:* Local CPU inference optimized with **INT8 Dynamic Quantization** (`torch.quantization.quantize_dynamic`), reducing memory overhead by 65%.
* **Voice Activity Gating (Silero VAD):** Slices incoming audio into 6.0s windows and filters out background silence and unvoiced room noise with a speech probability threshold ($p \ge 0.5$).
* **Non-Linear Quadratic Aggregation:** To defend against selective voice-splicing attacks (where only short phrases are cloned), chunks are aggregated quadratically:
  $$\text{Weight}_i = 1.0 + 2.0 \cdot (p_i)^2$$
  A single high-probability synthetic segment elevates the composite recording to an alert state.

### 2. Vision Forensic & Artifact Engine
Rather than relying on a single vision backbone, Truthy deploys **12 parallel diagnostic signals**:
* **Dual-CNN Ensembles (`core/image/models/cnn_detector.py`):**
  * `dima806/deepfake_vs_real_image_detection`: ViT fine-tuned for high-precision facial forgery detection with minimal false-positive rates on authentic human portraits.
  * `Ateeqq/ai-vs-human-image-detector`: Swin-transformer classifier trained on modern latent diffusion outputs (Midjourney, SDXL, DALL-E 3).
* **10 Classical Digital Signal Forensics (`core/image/analyzers/`):**
  * **2D Fast Fourier Transform (`frequency.py`):** Detects checkerboard upsampling artifacts and azimuthal radial spectral roll-off anomalies.
  * **Error Level Analysis (`ela.py`):** Re-compresses the image at 90% JPEG quality to detect local compression variance differences typical of digital composites.
  * **PRNU Sensor Noise Inconsistency (`noise.py`):** Uses wavelet high-pass residuals to flag unnatural, mathematically uniform synthetic noise distributions.
  * **Texture Homogeneity (`texture.py`):** Analyzes Gray-Level Co-occurrence Matrices (GLCM) and Local Binary Patterns (LBP).
  * **Optical Edge Gradients (`edge.py`):** Sobel filters measure fine-to-coarse edge sharpness ratios, distinguishing camera optical blur from diffusion smoothing.
  * **Color Covariance (`color.py`):** Checks channel correlation anomalies and chromatic aberration.
  * **Metadata Provenance (`metadata.py`):** Inspects EXIF headers and C2PA Content Credentials for generative software signatures.
  * **Compression & DCT (`compression.py`, `statistical.py`):** Quantization tables, Benford's law compliance, and coefficient skewness/kurtosis.
  * **Geometric Sanity (`geometry.py`):** Evaluates perspective vanishing points and shadow orientation consistency.
* **Forensic Manipulation Heatmap (`core/image/models/trufor.py`):** Synthesizes ELA, high-frequency energy, and noise residuals into a spatial RGB manipulation heatmap rendered as base64 PNG.
* **LLM Qualitative Arbitration (`core/image/judge.py`):** For borderline cases (probabilities between 35% and 65%), the system escalates the structured forensic telemetry to an LLM Judge (Gemini 2.0 Flash or Groq Llama 3.3 70B) for multi-factor synthesis.

---

## ⚡ Quick Start

### Prerequisites
* Python 3.10 or 3.11
* FFmpeg and libsndfile (`apt install ffmpeg libsndfile1` or `brew install ffmpeg libsndfile`)

### 1. Installation
```bash
git clone https://github.com/ace-ify/truthy.git
cd truthy

python -m venv venv
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install CPU-optimized PyTorch first, then requirements
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 2. Environment Configuration
```bash
cp .env.example .env
# Edit .env to supply HF_TOKEN, GEMINI_API_KEY, or API_KEYS if needed
```

### 3. Launch Development Server
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload
```
Navigate to:
* **Audio Forensic Dashboard:** `http://localhost:8080/`
* **Image Forensic Dashboard:** `http://localhost:8080/image`
* **Interactive OpenAPI Specs:** `http://localhost:8080/docs`

---

## 🐳 Containerized Deployment

Truthy is packaged with a production-ready, security-hardened `Dockerfile` featuring an automated healthcheck probe:

```bash
# Build Docker image
docker build -t truthy-engine .

# Run container
docker run -d \
  -p 8080:8080 \
  -e HF_TOKEN="your_hf_token_here" \
  -e JUDGE_MODE="weighted" \
  --name truthy \
  truthy-engine
```

### Deploying to Cloud Providers
* **Render:** Connected via `Procfile`. Set build command to install CPU wheels and `pip install -r requirements.txt`.
* **GCP Cloud Run / AWS ECS / Railway:** Deploy directly from the `Dockerfile`. Allocate $\ge 2\text{ GB}$ RAM to support local neural network caching.

---

## 🔌 API Reference

### 1. Voice Deepfake Detection (Base64)
`POST /api/voice-detection`

```bash
curl -X POST "http://localhost:8080/api/voice-detection" \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-api-key" \
  -d '{
    "language": "English",
    "audioFormat": "mp3",
    "audioBase64": "<BASE64_ENCODED_AUDIO>"
  }'
```

**Response Payload:**
```json
{
  "status": "success",
  "language": "English",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.94,
  "explanation": "Strong synthetic voice patterns detected. Audio shows clear signs of AI generation with unnatural pitch consistency and robotic speech artifacts in English."
}
```

### 2. Image Forensic Analysis (File Upload)
`POST /api/image-analyze`

```bash
curl -X POST "http://localhost:8080/api/image-analyze" \
  -F "file=@sample.jpg" \
  -F "mode=standard"
```

**Response Payload:**
```json
{
  "status": "success",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.88,
  "confidence": "High",
  "explanation": "Both CNN models agree AI-generated. Frequency spectrum exhibits unnatural grid-like harmonics characteristic of latent diffusion upsampling.",
  "generatorGuess": "Stable Diffusion / Midjourney",
  "signalBreakdown": [
    {
      "analyzer_name": "cnn_detector",
      "display_name": "CNN Classifier",
      "signal_score": 0.92,
      "confidence": 0.90,
      "reasoning": "Both CNNs agree: AI-generated"
    },
    {
      "analyzer_name": "frequency",
      "display_name": "Frequency Analysis",
      "signal_score": 0.84,
      "confidence": 0.75,
      "reasoning": "Radial energy roll-off exhibits uncharacteristic high-frequency spikes"
    }
  ],
  "heatmapBase64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
}
```

### 3. Server-Sent Events (SSE) Live Stream
`POST /api/image-stream`  
Streams real-time diagnostic outputs as individual analyzer threads complete, culminating in the composite verdict.

---

## 📊 Comparative Performance Matrix

| Feature | Monolithic Deepfake Detectors | Commercial Closed APIs (Hive, Sensity) | **Truthy Hybrid Engine** |
| :--- | :---: | :---: | :---: |
| **Modalities Covered** | Single (Audio OR Vision) | Multiple | **Unified (Audio + Vision + Provenance)** |
| **Detection Methodology** | Single CNN / ViT Backbone | Proprietary Black-Box | **Dual-CNN + 10x DSP Signal Telemetry** |
| **Silence / Noise Immunity** | ❌ Diluted by silence | ⚠️ Opaque | **✅ Silero VAD Speech Gating** |
| **Spatial Localization** | ❌ None | ❌ None | **✅ Pixel-Level Forensic Heatmaps** |
| **Explainability** | ❌ Raw float score | ❌ Opaque score | **✅ Multi-Signal Diagnostics & LLM Rationale** |
| **Spliced Audio Detection**| ❌ Averaged out | ⚠️ Inconsistent | **✅ Quadratic Anomaly Weighting** |
| **Self-Hostable** | ✅ Yes | ❌ Closed SaaS | **✅ Fully Open & Containerized (Docker)** |

---

## 🛠️ Tech Stack
* **Backend Framework:** FastAPI, Uvicorn, SlowAPI (rate limiting)
* **ML & Deep Learning:** PyTorch, TorchAudio, Hugging Face Transformers
* **Signal Processing & Computer Vision:** OpenCV (headless), NumPy, SciPy, Librosa, NoiseReduce, Scikit-Image, Pillow, PyExif
* **AI Arbitration:** Google GenAI (Gemini 2.0 Flash), Groq (Llama 3.3 70B)
* **Infrastructure:** Docker, Render, Hugging Face Inference API

---

## 📜 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
