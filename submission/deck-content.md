# Truthy - Round 1 deck content (build-your-own base)

Track 3: Deepfake Detection for Financial Communications.
This is tight, ready-to-place copy + speaker notes + layout hints for each slide. Build the
visuals yourself (Canva / PowerPoint / Slides / Gamma). Format: 10 pitch slides + up to 3
architecture pages, in ONE PDF (the 3 architecture pages are separate from the 10, not counted in them).

Real numbers to use (do not inflate): calibrated primary detector - leave-one-out accuracy 87%,
ECE 0.047, on n=23 confident image samples; 62% of a hard mixed set routed to human review.
Primary models: dima806 + Ateeqq (image), wav2vec2-xlsr (voice). No em-dashes anywhere.

---

## Slide 1 - Title
**On slide:**
- Truthy
- Verify before you pay.
- One line: A verifier for financial communications. Drop in a suspicious voice note, video clip, or payment screenshot and get a plain verdict, the evidence, and a safe next step.
- Team name + members. Track 3.

**Speaker notes:** We built the check that happens in the last second before money leaves an account.
**Layout:** Big wordmark, one-line promise, quiet. No stock imagery needed.

---

## Slide 2 - The user and the harmful moment
**On slide:**
- Headline: The most expensive second in Indian digital finance.
- Primary user: the person at the payment decision (retail UPI user, SME accountant, or bank ops reviewer).
- The moment: an urgent message arrives right before a UPI transfer or wire.
- Three cheap-to-fake attacks: cloned-voice WhatsApp orders, deepfake video calls, forged payment screenshots / KYC selfies.

**Speaker notes:** The threat is not "somewhere on the internet." It is one message, one decision, one person under time pressure.
**Layout:** One headline + three labelled attack tiles. India-specific framing.

---

## Slide 3 - Before and after
**On slide:**
- Headline: From "panic and pay" to "check, then act".
- Today: urgent message, pressure, pay first and doubt later. No fast way to check. A raw "92% fake" score means nothing. No record, no escalation.
- With Truthy: drop the file, verdict in seconds, plain "do not act, confirm on a known number", evidence you can show your bank, auto-escalation to a human when the model is unsure.

**Speaker notes:** The differentiator is not a detector score. It is a decision plus evidence plus a safe action plus a case record.
**Layout:** Two columns, today vs with-Truthy. Red-tinted vs green-tinted.

---

## Slide 4 - Product walkthrough
**On slide:**
- Headline: One report a non-technical person can act on.
- Verdict + risk (icon, word, colour, never colour alone).
- "What you should do" - the safe action in plain language.
- Primary vs supporting evidence, clearly separated.
- Provenance panel (C2PA / EXIF). Case ID + SHA-256, exportable report, escalate.

**Speaker notes:** Point at the screenshot. The verdict reads in one glance; the evidence is there when you want it.
**Layout:** Real product screenshot (use `submission/assets/image-page-redesign.png` or `truthy-report-ai-image.png`) + 4-5 short callouts.

---

## Slide 5 - Detection approach (the honesty core)
**On slide:**
- Headline: The model decides. Heuristics only nudge.
- Primary, model-backed: dual-CNN (image), wav2vec2 (voice), mapped to a calibrated probability.
- Supporting, heuristic: ELA, frequency, noise, phase, rPPG, lip-sync. Bounded nudge of at most 0.15.
- The one rule: a supporting signal can never flip a confident model verdict.
- When the model is not confident, the result is Inconclusive and routes to human review. We do not guess.

**Speaker notes:** This single rule is why a confident wrong answer is hard to produce. It is the safety design, and it is what most "AI detector" demos get wrong.
**Layout:** Simple left-to-right flow: primary + supporting -> calibrated verdict / escalate.

---
<!-- MORE -->

## Slide 6 - Provenance and the case record
**On slide:**
- Headline: The strongest evidence, and a chain of custody.
- C2PA / Content Credentials verified cryptographically. A signed origin claim beats any pixel heuristic.
- EXIF + JPEG quantization: camera-capture chain vs AI-software marker vs messaging recompression.
- Precedence rule: verified provenance sets the headline; forgeable EXIF only fills an Inconclusive gap.
- Every check becomes a case: ID, SHA-256 of the media, verdict, timestamp. The media itself is never stored.
- Exportable evidence report to forward to a bank or fraud team.

**Speaker notes:** Provenance is where we go beyond "looks fake." A verified Content Credential is proof of origin, not a guess.
**Layout:** Left bullets + right screenshot of the evidence report (`truthy-evidence-report.png`).

---

## Slide 7 - Data, validation, assumptions
**On slide:**
- Headline: Real numbers, and what they do and do not mean.
- 87% leave-one-out accuracy of the calibrated primary on confident cases.
- ECE 0.047 (confidence tracks reality), on n=23.
- 62% of a hard mixed set routed to human review instead of guessed. The escalation rate is the safety feature.
- Data: synthetic / public / consented samples only, including WhatsApp-compressed portraits.
- Assumption stated plainly: n is small. This calibrates confidence, it is not a benchmark accuracy claim.

**Speaker notes:** We are showing our real, modest numbers and our limitation out loud. That honesty is the point of a trust product.
**Layout:** Three stat tiles + a short assumptions list.

---

## Slide 8 - Safety, privacy, responsible AI
**On slide:**
- Headline: Built so that being wrong cannot cause harm.
- No auto-blocking or accusations on a single weak signal. Truthy advises, the human acts.
- Human review path for every escalated or low-confidence case, gated by a reviewer key that fails closed.
- Consent: the uploader confirms authorisation. Enforced server-side.
- Privacy: media processed transiently and never stored. Only metadata and a SHA-256 hash are kept.
- Adversarial: heuristics cannot drive the verdict, so poisoning them cannot flip it. Provenance forgery is capped to a nudge.

**Speaker notes:** Map each point to a rubric line: false positives, human review, consent, misuse, privacy.
**Layout:** Bullets + screenshot of the review queue (`truthy-review-queue.png`).

---

## Slide 9 - Adoption, integration, scale
**On slide:**
- Headline: A drop-in verification step, three ways to ship.
- Consumer PWA: forward a suspicious voice note or screenshot from WhatsApp, get a verdict. No install.
- Bank / wallet SDK: call the verify endpoint in-app at the confirm-payment step for flagged flows.
- PSP / ops console: fraud-ops review queue with exportable evidence reports for dispute files.
- Owner: bank or PSP fraud-risk team runs the queue; platform trust team owns thresholds and calibration refresh.
- Rollout: internal ops assist, then high-risk in-app flows, then consumer share-sheet.
- Latency: image roughly 2 to 8 seconds on CPU; single worker to avoid model duplication; scale by replicas.

**Speaker notes:** Emphasise it slots into an existing payment step; it is not a separate destination the user has to remember.
**Layout:** Three integration tiles + a short owner/rollout/latency strip.

---

## Slide 10 - Feasibility and the 48-hour plan
**On slide:**
- Headline: A narrow, demonstrable MVP. Most of it already runs.
- Already working: unified verify UI for image / voice / video; calibrated dual-CNN + wav2vec2; C2PA + EXIF provenance; case ledger; evidence report; review queue.
- 48-hour targets: live demo of a cloned-voice UPI note, a forged screenshot, and a deepfake clip; WhatsApp share-sheet front-end; expand and refit calibration per attack type with published ROC; reviewer analytics + a webhook alert to a mock bank ops channel.

**Speaker notes:** We are not starting from zero at the finale. We are extending a working system and proving it on the three attack types end to end.
**Layout:** Two columns, already-working vs 48-hour-targets.

---

# Architecture pages (up to 3, separate from the 10 slides)

## Architecture 1 - End-to-end system (one diagram)
Draw a single top-to-bottom flow:
1. Ingest (POST /api/verify): voice note / video clip / image. Consent required. SHA-256 at entry. Temp file, deleted after.
2. Three pipelines: Voice (librosa 16k, Silero VAD, wav2vec2 = primary). Image (dual-CNN = primary, plus 11 DSP forensics + attention heatmap = supporting). Video (frames -> rPPG + lip-sync = supporting; audio -> wav2vec2 = primary).
3. Tiered fusion + calibration (Platt scaling, primary drives, supporting nudge, uncertain -> Inconclusive) AND Provenance (C2PA + EXIF + quant; verified provenance outranks pixels).
4. Decision envelope: verdict, risk, calibrated confidence, plain recommendation, signals, provenance.
5. Case ledger (SQLite, metadata + hash, media never stored) -> evidence report + human review queue.

**Layout:** One legible boxes-and-arrows diagram, top to bottom. Label primary vs supporting.

## Architecture 2 - Inputs, models, storage, interfaces
- Inputs / consent / provenance: media up to 20-50 MB; consent flag mandatory; provenance from C2PA manifest, EXIF (make/model/datetime/GPS/lens), JPEG DQT; SHA-256 as the evidentiary anchor.
- Models / rules: primary CNNs + wav2vec2 (HF, pinned; CPU/INT8 or HF Inference API); Platt calibrator (scipy, persisted calibration.json); supporting numpy/scipy DSP + rPPG + lip-sync with bounded influence.
- Storage / boundaries: SQLite case ledger (metadata + hash + verdict + review state); no media at rest; temp files unlinked.
- Interfaces / outputs: REST (verify, case, report, escalate, review), SSE for live image; JSON envelope + printable HTML evidence report; unified verify UI + analyst review queue.

**Layout:** Four quadrants or four short blocks.

## Architecture 3 - Flow, safeguards, validation
- Decision and escalation flow: confident primary -> verdict + risk + recommendation; uncertain or no-primary -> Inconclusive -> auto-queue for review; reviewer confirms or overrides with a note.
- Security and privacy: review endpoints require a reviewer key (no same-origin bypass), fail closed; transient media, hash-only retention; consent-gated ingest; tightened CORS on mutations; rate-limited public endpoints.
- Adversarial risk: heuristics cannot drive the verdict; provenance forgery capped to a nudge; unknown or ambiguous defaults to escalation.
- Performance and validation: image roughly 2 to 8 seconds on CPU; single worker; LOO accuracy + ECE today, ROC + per-attack sets next; calibration refresh as labelled data grows, with disclosed sample size.

**Layout:** Four short blocks. Footer line: synthetic/public/consented data only; decision-support, not legal proof.

---

## Notes for whoever builds the deck
- Keep one accent colour plus the red/amber/green risk semantics. Avoid glows and gradient soup.
- Use the real screenshots in `submission/assets/`.
- There is a rough HTML draft at `submission/deck.html` (renders to a PDF) if you want a starting skeleton, but treat the copy here as the source of truth.
- Reviewers score the PDF only, so a reviewer must understand it with no live demo. Lead every slide with the point, not the mechanism.

