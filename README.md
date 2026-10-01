# Truthy — Verify before you pay

**Track 3 (Deepfake Detection for Financial Communications) · Amazon AI Cyber Security Hackathon**

Truthy is a **multimodal financial-communications verifier** built around one moment: someone is
about to authorize a UPI/wire transfer and receives a suspicious **voice note, video clip, or
payment screenshot/document**. They drop it into Truthy and get a **plain-language verdict, the
evidence behind it, a safe next step, and an auditable case record** — not just a raw score.

## What makes it trustworthy (the design, honestly)

Detection is built on an **honesty-tiering** rule, because a confident-looking wrong answer is the
real danger in a financial-safety product:

- **Primary (model-backed) signals** drive the verdict: a dual-CNN image classifier
  (`dima806/deepfake_vs_real_image_detection` + `Ateeqq/ai-vs-human-image-detector`) and a
  wav2vec2 voice-clone model (`Gustking/wav2vec2-large-xlsr-deepfake-audio-classification`).
- **Supporting (heuristic) signals** — classic DSP forensics (ELA, FFT, noise, phase, …), the
  illustrative attention heatmap, rPPG pulse, and a lip-sync heuristic — may nudge confidence by at
  most **±0.15** and **can never flip a confident model verdict**.
- When the primary model is **not confident, the system does not guess** — it returns
  **Inconclusive** and routes the case to **human review**.

**Confidence is calibrated, not fabricated.** A Platt-scaling calibrator (`core/image/calibration.py`,
fit by `tests/fit_calibration.py`) maps the primary score to a real probability. On our labeled
sample set the calibrated primary is **~87% accurate (leave-one-out), ECE 0.047 on n=23 confident
cases**, and **~62% of a hard mixed set is routed to review** rather than decided — that escalation
rate is the safety feature. If no calibration file is present, confidence is labelled *uncalibrated*
and capped — we never claim a number we didn't fit.

## Provenance (the strongest evidence)

`core/provenance/` verifies **C2PA / Content Credentials** (cryptographically, via `c2pa`), reads
structured **EXIF** with `piexif`, and inspects **JPEG quantization** tables. A *verified* Content
Credential or a coherent camera-capture EXIF chain outranks any pixel heuristic (the precedence
rule in `provenance.apply_precedence`). Falls back to a byte-scan ("present but unverified") when
the manifest can't be validated — never a fake "verified".

## Governance & evidence (Track-3 accountability)

- **Case ledger** (`core/governance/store.py`, stdlib `sqlite3`): every check gets a case ID,
  a **SHA-256 of the media**, verdict, and review status. **The media itself is never stored** —
  only metadata + hash (transient-processing privacy posture).
- **Exportable evidence report**: `GET /api/case/{id}/report` renders a printable HTML report
  (browser "Save as PDF") with chain of custody, evidence, provenance, and disclosures.
- **Human review / escalation**: escalated and low-confidence cases queue for an analyst at
  `/review` (gated by `REVIEWER_KEY`, fails closed) who can confirm or override with a note.
- **Consent**: uploads require an authorization checkbox; enforced server-side too.

## API

| Endpoint | Purpose |
|---|---|
| `POST /api/verify` | Unified: voice/video/image → one report envelope + case ID |
| `GET /api/case/{id}` / `/report` | Case JSON / printable HTML evidence report |
| `POST /api/case/{id}/escalate` | Queue for human review |
| `GET /api/review`, `POST /api/case/{id}/review` | Reviewer queue + decision (needs `x-reviewer-key`) |
| `POST /api/voice-detection` · `/image-detection` · `/video-detection` | Base64 API (legacy contract) |
| `POST /api/image-stream` | Live SSE image analysis |

UI: `/` (unified verify), `/review` (analyst queue). `/image`, `/video`, `/audio` deep-link into the
right tab.

## Run

```bash
python -m venv .venv && . .venv/Scripts/activate      # (Linux/mac: source .venv/bin/activate)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python tests/fit_calibration.py                        # fit calibration.json (downloads CNN models once)
uvicorn api.main:app --host 0.0.0.0 --port 8080 --workers 1
```

Docker: `docker build -t truthy . && docker run -p 8080:8080 truthy` (single worker — each worker
loads its own ~2GB model copy; scale with replicas, not in-process workers). Set `HF_TOKEN` to run
the audio model via the HF Inference API and keep it out of local RAM.

## Limitations (stated plainly)

- Calibration is on a **small validation set (n=23 confident)** — it calibrates *confidence* and
  sanity-checks thresholds; it is **not** a benchmark accuracy claim. Disclosed everywhere it's used.
- **rPPG and lip-sync are experimental supporting signals**, not reliable deepfake detectors; a real
  audio-visual sync model (SyncNet-class) is future work. On video, the **voice-clone check is the
  only model-backed signal**, so a real-voice + face-swapped video is escalated, not cleared.
- Truthy is **decision-support, not legal proof**; a detector score is not a verdict on a person.

## Tech
FastAPI · PyTorch (CPU) · Transformers · OpenCV · scikit-image · scipy · Pillow/piexif · c2pa · SQLite.
