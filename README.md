<p align="center">
  <img src="https://see.fontimg.com/api/rf5/rv0aB/ZTRlNjgyYThiNTRjNDE1ZWFiYzliZmY5OGI5MDhhM2Yub3Rm/dHJ1dGh5/teknaf-regular.png?r=fs&h=81&w=1250&fg=3b82f6&bg=000000&s=65" alt="Truthy Logo" height="60">
</p>

# Truthy - AI Voice Detection

Detect AI-generated voices and deepfake audio. Built for a hackathon.

Uses WavLM + Silero VAD under the hood to classify audio as human or AI-generated, with per-chunk analysis and confidence scoring.

## What it does

- Takes in audio (file upload or Base64 via API)
- Runs Voice Activity Detection to find speech segments
- Classifies each segment using a fine-tuned WavLM model
- Returns an overall verdict with confidence score

Supports English, Hindi, Tamil, Telugu, Malayalam.

## Quick start

```bash
git clone https://github.com/ace-ify/truthy.git
cd truthy

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
python -m uvicorn api.main:app --reload --port 8000
```

Then open `http://localhost:8000`.

You'll need ~4GB RAM for model loading.

## API

All external API calls need an `x-api-key` header. Requests from the same origin (the website itself) skip auth.

### `POST /api/voice-detection`

Main endpoint. Accepts Base64-encoded audio.

```json
{
  "language": "English",
  "audioFormat": "mp3",
  "audioBase64": "..."
}
```

Response:
```json
{
  "status": "success",
  "language": "English",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.87,
  "explanation": "Strong synthetic voice patterns detected..."
}
```

### `POST /api/analyze`

File upload endpoint (legacy, used for testing).

### `GET /api/health`

Returns model status and device info.

### `GET /api/info`

Lists supported languages and available endpoints.

### Errors

All errors return `{"status": "error", "message": "..."}` with appropriate HTTP status codes (400, 401, 429, 500).

## Project structure

```
truthy/
├── api/
│   ├── main.py            # FastAPI app, routes, auth
│   └── models.py          # Pydantic request/response schemas
├── core/
│   ├── detector.py        # WavLM deepfake classifier (HF API + local fallback)
│   ├── vad.py             # Silero VAD wrapper
│   ├── audio_processor.py # Loading, resampling, noise reduction
│   └── aggregator.py      # Combines chunk results into final verdict
├── static/                # Frontend (HTML/CSS/JS)
├── config.py              # Settings and thresholds
├── requirements.txt
├── Dockerfile
└── Procfile
```

## Tech stack

**Backend:** FastAPI, PyTorch, WavLM, Silero VAD, librosa, noisereduce

**Frontend:** HTML/CSS/JS, TailwindCSS, Lucide Icons

**Infra:** Render (deployment), HuggingFace (model hosting)

## Deployment

Deployed on Render. Set these env vars:

| Variable | Value |
|----------|-------|
| `API_KEYS` | comma-separated API keys |
| `PYTHONUNBUFFERED` | `1` |
| `HF_TOKEN` | your HuggingFace token (optional, enables cloud inference) |

Start command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`

## How detection works

Audio gets split into 6-second chunks. Each chunk goes through:

1. **VAD** — Silero checks if there's actual speech in the chunk
2. **Classification** — Speech chunks get scored by a WavLM-based model (fine-tuned for deepfake detection)
3. **Aggregation** — Chunk scores are combined using a max-weighted average. If any chunk has a strong AI signal (>70%), it pulls the overall score up.

Threshold is 0.5 — above that it's flagged as AI.

## Roadmap

- More Indian languages (Kannada, Bengali, Gujarati)
- Real-time streaming detection
- Browser extension
- Mobile SDK

## License

MIT

---

Built for hackathon by the Truthy team.
