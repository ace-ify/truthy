# Truthy -- Hackathon PPT Guideline

> **HACK X VID-YOUTH 2026** | Max 7 slides (including title) | Submit as PDF
> Keep it points/diagrams/infographics -- NO paragraphs.

---

## SLIDE 1 -- Team Details & Members

**Content:**

- **Team Name:** [Your Team Name]
- Member 1 -- Name, Role (e.g., ML Engineer), Key Skills (PyTorch, Audio DSP)
- Member 2 -- Name, Role (e.g., Backend Dev), Key Skills (FastAPI, Python)
- Member 3 -- Name, Role (e.g., Frontend + DevOps), Key Skills (HTML/CSS/JS, Docker)
- Contact info (email / phone)

**Visual Idea:**
- Horizontal card layout -- one card per member with avatar placeholder, name, role, skill tags
- Color-coded role badges (ML = blue, Backend = green, Frontend = orange)

---

## SLIDE 2 -- Problem Statement & Proposed Solution

### Problem (Left Column)

Use these bullet points:

- **Deepfake crisis is exploding** -- AI-generated voices used for financial fraud, political manipulation, and identity theft
- **AI images flood the internet** -- Stable Diffusion, DALL-E, Midjourney produce photorealistic fakes indistinguishable to the human eye
- **Trust erosion** -- 96% of deepfake videos are non-consensual; $25B+ in fraud losses attributed to synthetic media (2024)
- **No single tool covers both** -- existing detectors handle only audio OR only images, not both in one platform
- **Detection tools are inaccessible** -- require ML expertise, command-line tools, or expensive enterprise subscriptions

### Solution (Right Column)

- **Truthy** -- A unified AI-generated content detection platform
- Detects **deepfake audio** AND **AI-generated images** in a single web app
- **14 parallel forensic + ML analyzers** for images (frequency, noise, ELA, CNN, CLIP, DIRE, etc.)
- **WavLM-based voice classifier** with VAD-filtered chunking for audio
- **3-tier analysis** -- Quick (~2s), Standard (~10s), Thorough (~25s)
- **LLM-powered reasoning** (Gemini/Groq) for ambiguous edge cases
- **Zero expertise needed** -- simple upload-and-detect web interface

**Visual Ideas:**

```
DIAGRAM: Problem-Solution Split Layout

+---------------------------+---------------------------+
|        THE PROBLEM        |       OUR SOLUTION        |
|                           |                           |
|  [Icon: Warning]          |  [Icon: Shield/Check]     |
|  Deepfake voice fraud     |  WavLM voice detection    |
|                           |                           |
|  [Icon: Image]            |  [Icon: Magnifier]        |
|  AI-generated images      |  14 forensic analyzers    |
|                           |                           |
|  [Icon: Broken trust]     |  [Icon: Brain]            |
|  Eroding public trust     |  LLM-powered reasoning    |
|                           |                           |
|  [Icon: Lock]             |  [Icon: Globe]            |
|  Tools are inaccessible   |  Simple web interface     |
+---------------------------+---------------------------+
```

- Use a **before/after** or **problem arrow solution** flow
- Big stat callouts: "96% of deepfakes are non-consensual", "$25B+ fraud losses"

---

## SLIDE 3 -- Technical Approach

This is your most important slide. Pack it with architecture, not paragraphs.

### Tech Stack (Top Strip)

| Layer | Tech |
|-------|------|
| Backend | FastAPI 2.0, Python 3.11+, uvicorn |
| Audio ML | WavLM (wav2vec2-large-xlsr), Silero VAD |
| Image ML | CNN ResNet, CLIP (ViT-L/14), DIRE (SD-VAE), TruFor |
| Image Forensics | FFT Frequency, ELA, Noise, Texture (LBP), Edge (Sobel), Color, Metadata, Compression (DCT), Statistical (Benford), Geometry |
| LLM Judge | Gemini 2.0 Flash, Llama 3.3 70B (Groq) |
| Frontend | Vanilla JS + TailwindCSS |
| Infra | Docker, Render, SSE Streaming |

### Flow 1: Voice Deepfake Detection Pipeline

```
DIAGRAM: Audio Pipeline (Horizontal Flow)

[Audio Upload] --> [Resample 16kHz] --> [Noise Reduction] --> [6s Chunking]
                                         (spectral gating)
        |
        v
[Silero VAD] ----filter----> [Speech Chunks Only]
        |
        v
[WavLM Classifier] --> per-chunk AI probability
        |
        v
[Weighted Aggregation] --> max*0.7 + avg*0.3
        |
        v
[Verdict: AI Generated / Human Voice]
   + Confidence (High/Medium/Low)
```

### Flow 2: Image Detection Pipeline

```
DIAGRAM: Image Pipeline (Vertical Parallel Architecture)

                    [Image Upload]
                         |
                    [Preprocess]
                 (RGB, Resize, EXIF)
                         |
            +------------+------------+
            |                         |
     [10 Forensic               [4 ML Model
      Analyzers]                 Detectors]
            |                         |
   +--+--+--+--+--+         +--+--+--+--+
   |F |N |E |T |...|        |CNN|CLIP|DIRE|TruFor|
   +--+--+--+--+--+         +--+--+--+--+
            |                         |
            +------------+------------+
                         |
                   [Judge System]
              weighted / llm / hybrid
                         |
               [Verdict + Confidence
                + Signal Breakdown
                + Heatmap]
```

### Flow 3: Judge Decision System

```
DIAGRAM: Judge Logic (Decision Tree)

[14 Analyzer Results]
        |
   [Weighted Ensemble]
   score = SUM(score * weight * confidence)
             / SUM(weight * confidence)
        |
   score in [0.35 - 0.65]?
      /        \
    NO          YES (borderline)
    |            |
  [Return      [Escalate to LLM]
   Verdict]     Gemini / Groq
                    |
              [LLM Reasoning]
              temp=0.1, 500 tokens
                    |
              [Final Verdict]
```

**Visual Ideas for this slide:**

- **Architecture diagram** showing the two parallel pipelines (audio + image) converging
- Use **colored blocks** -- blue for audio path, green for image path, gold for judge
- **Analyzer weight bar chart** showing the 10 analyzers ranked by weight (1.5 down to 0.5)
- **Icon grid** for the 14 analyzers (2 rows x 7 columns) with small icons and labels
- A small **SSE streaming timeline** showing real-time results appearing one by one

---

## SLIDE 4 -- Feasibility and Viability

### Feasibility Analysis

- **All models are open-source** -- WavLM, CLIP, ResNet, Silero VAD (HuggingFace ecosystem)
- **Dual inference strategy** -- HuggingFace API (primary) + local fallback with Int8 quantization
- **Runs on CPU** -- no GPU required; dynamic quantization reduces memory to ~4GB RAM
- **Dockerized** -- single `docker build && docker run` deployment
- **Already functional** -- working prototype with 8 API endpoints, SSE streaming, full frontend

### Challenges and Risks

| Challenge | Mitigation Strategy |
|-----------|-------------------|
| Model accuracy on novel AI generators | Ensemble of 14 analyzers reduces single-point failure; LLM judge adds reasoning layer |
| Processing time for thorough analysis | 3-tier mode system (quick=2s, standard=10s, thorough=25s); SSE streaming shows progress |
| HuggingFace API rate limits / downtime | Automatic fallback to local quantized models |
| Large model downloads (~2GB first run) | Lazy loading + model caching in `models/` directory |
| Adversarial attacks on detectors | Multi-signal approach: forensic + ML + LLM makes it hard to fool all 14 analyzers simultaneously |
| Scaling for high traffic | Rate limiting (100/min voice, 60/min image); ThreadPoolExecutor with max 8 parallel analyzers |

### Why It Works

- **Redundancy** -- If one analyzer fails, 13 others still contribute
- **Conflict detection** -- If analyzers disagree (std > 0.25), system flags "Inconclusive" instead of guessing
- **Graceful degradation** -- Every ML model has an API-to-local fallback path

**Visual Ideas:**

```
DIAGRAM: Feasibility Radar / Risk Matrix

         High Impact
              |
  [Model      |      [API
   Accuracy]  |    Downtime]
              |
Low Likelihood ----+---- High Likelihood
              |
  [Adversarial|    [Processing
   Attacks]   |     Time]
              |
         Low Impact

Each risk has an arrow pointing to its mitigation
```

- **Traffic light indicators** (green/yellow/red) next to each challenge
- **Fallback cascade diagram**: HF API --> Local Model --> LLM Judge --> Weighted Ensemble

---

## SLIDE 5 -- Impact and Benefits

### Target Audience

- **Journalists & fact-checkers** -- verify media authenticity before publishing
- **Social media platforms** -- flag AI-generated content at upload
- **Law enforcement** -- detect voice deepfakes in evidence
- **Banks & financial institutions** -- prevent voice-clone fraud in KYC/auth
- **General public** -- empower anyone to verify suspicious content

### Impact Metrics (Projected)

| Metric | Value |
|--------|-------|
| Detection modalities | 2 (audio + image) |
| Forensic signals analyzed per image | 14 (10 forensic + 4 ML) |
| Languages supported | 5 (EN, HI, TA, TE, ML) |
| Analysis modes | 3 (quick/standard/thorough) |
| Fastest detection | ~2 seconds |
| API endpoints | 8 |

### Benefits Breakdown

- **Social** -- Combats misinformation, protects against identity theft, preserves trust in media
- **Economic** -- Reduces fraud losses ($25B+ annually); saves manual fact-checking time
- **Technological** -- Open-source ML stack; reproducible and extensible
- **Accessibility** -- Web-based, no ML expertise needed; supports 5 Indian languages
- **Transparency** -- Signal breakdown shows WHY the verdict was reached (not a black box)

**Visual Ideas:**

```
DIAGRAM: Impact Ripple / Concentric Circles

            +---------------------------+
            |        SOCIETY            |
            |   +-------------------+   |
            |   |    INDUSTRY       |   |
            |   |  +-------------+  |   |
            |   |  |   USER      |  |   |
            |   |  | Upload -->  |  |   |
            |   |  | Detect -->  |  |   |
            |   |  | Trust       |  |   |
            |   |  +-------------+  |   |
            |   | Banks, Media,     |   |
            |   | Platforms         |   |
            |   +-------------------+   |
            | Reduced fraud,            |
            | Restored trust            |
            +---------------------------+
```

- **Icon + stat callout** grid: "14 Signals", "5 Languages", "2s Detection", "2 Modalities"
- **Before Truthy / After Truthy** comparison strip

---

## SLIDE 6 -- Future Scope / Next Steps

### Short-term (Post-hackathon, 1-3 months)

- **Deepfake video detection** -- frame-by-frame + temporal consistency analysis
- **Browser extension** -- right-click any image/audio to verify
- **Batch processing API** -- bulk upload for newsroom workflows
- **Accuracy benchmarking** -- test against ASVspoof, FakeAVCeleb, GenImage datasets
- **GPU acceleration** -- CUDA support for sub-second thorough analysis

### Medium-term (3-6 months)

- **Text AI detection** -- GPTZero-like analysis for AI-written articles
- **Blockchain provenance** -- content authenticity certificates on-chain
- **Mobile app** (React Native) -- detect deepfakes on-the-go
- **Webhook integrations** -- auto-scan uploads on WhatsApp Business, Telegram bots
- **Fine-tuning** on Indian-language deepfakes (Hindi, Tamil, Telugu voice clones)

### Long-term Vision

- **Truthy as an API layer** -- become the "Stripe for content authenticity"
- **Real-time detection** -- streaming audio/video analysis for live broadcasts
- **Coalition for Authentic Media** -- open standard for media provenance (like C2PA)
- **Government integration** -- election commission tool for political ad verification

**Visual Ideas:**

```
DIAGRAM: Roadmap Timeline (Horizontal)

Now          1-3 months       3-6 months        1 year+
 |               |                |                |
 *-----[v1]-----*---[v2]---------*---[v3]---------*
 |               |                |                |
 Audio+Image     Video Detection  Text Detection   Real-time
 Detection       Browser Plugin   Mobile App       Streaming
 Web App         Batch API        Blockchain       API Platform
                 GPU Support      Webhooks         Govt Integration
```

- **Layered cake / building blocks** showing features stacking over time
- **3 milestone cards** with icons (rocket, puzzle piece, globe)

---

## SLIDE 7 -- Conclusion

### Summary (3 key points)

1. **Truthy is a dual-modality AI detection platform** -- detects deepfake voices AND AI-generated images in one unified tool
2. **14 parallel forensic + ML analyzers** with a hybrid judge system (weighted ensemble + LLM reasoning) deliver robust, explainable verdicts
3. **Production-ready prototype** -- FastAPI backend, real-time SSE streaming, Docker-deployable, 5 Indian languages supported

### Key Takeaways

- Multi-signal ensemble approach is **more resilient** than any single detector
- **Transparency** -- users see exactly which signals triggered the verdict
- **Accessibility** -- simple web upload, no ML expertise required

### Why Our Approach Stands Out

| Others | Truthy |
|--------|--------|
| Single modality (audio OR image) | **Both** audio AND image |
| Black-box verdict | **14-signal breakdown** with reasoning |
| Single model | **Ensemble of 14** with LLM fallback |
| English only | **5 Indian languages** |
| Cloud-only | **API + local fallback** (works offline) |
| Slow batch processing | **Real-time SSE streaming** |

**Visual Ideas:**

```
DIAGRAM: Comparison Table (styled as cards)

+------------------+      +------------------+
| EXISTING TOOLS   |  VS  |     TRUTHY       |
|                  |      |                  |
| Single modality  |      | Dual modality    |
| Black box        |      | 14-signal expln  |
| One model        |      | 14 analyzers     |
| English only     |      | 5 languages      |
| Cloud dependent  |      | API + local      |
+------------------+      +------------------+
```

- End with a **bold tagline**: *"Truthy -- Because seeing shouldn't mean believing."*
- Or: *"14 signals. 2 modalities. 1 truth."*

---

## General Visual & Design Recommendations

### Color Palette (Topic-Specific)

| Role | Color | Hex | Why |
|------|-------|-----|-----|
| Primary | Deep Navy | `#1E2761` | Trust, security, technology |
| Secondary | Electric Teal | `#028090` | AI, detection, clarity |
| Accent | Coral Red | `#F96167` | Danger (deepfakes), urgency |
| Background | Off-white | `#F5F5F5` | Clean, readable |
| Text | Charcoal | `#2D2D2D` | High contrast |

### Diagrams to Create

1. **System Architecture** (Slide 3) -- The most important visual. Two parallel pipelines converging at the judge.
2. **Audio Pipeline Flow** (Slide 3) -- Horizontal 6-step pipeline with icons at each stage.
3. **Image Analyzer Grid** (Slide 3) -- 2x7 icon grid showing all 14 analyzers with weight labels.
4. **Risk Mitigation Matrix** (Slide 4) -- 2x2 quadrant or table with traffic light colors.
5. **Impact Circles** (Slide 5) -- Concentric circles from user to society.
6. **Roadmap Timeline** (Slide 6) -- Horizontal milestone strip.
7. **Comparison Table** (Slide 7) -- Side-by-side "Others vs Truthy" with checkmarks/crosses.

### Tools for Creating Diagrams

- **Excalidraw** (excalidraw.com) -- hand-drawn style, great for architecture diagrams
- **Mermaid.js** -- code-to-diagram, embed in markdown
- **draw.io** (app.diagrams.net) -- flowcharts and system diagrams
- **Canva** -- quick infographics
- **Figma** -- polished custom visuals

### Key Numbers to Highlight (Big & Bold)

- **14** forensic + ML analyzers
- **2** modalities (audio + image)
- **5** Indian languages
- **~2s** fastest detection
- **3** analysis modes
- **8** API endpoints

---

## PPT Rules Reminder

1. Max **7 slides** (including title slide)
2. **No paragraphs** -- use points, diagrams, infographics, images
3. **Precise and easy to understand**
4. **Unique and novel** idea
5. Save as **PDF** for upload
