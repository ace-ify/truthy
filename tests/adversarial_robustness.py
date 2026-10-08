"""
Adversarial / robustness probe ("evolving manipulation methods").

Applies common perturbations an adversary (or just messaging-app recompression) would do
to an image, and measures how often the verdict FLIPS. The thesis of the honesty design:
because the calibrated model drives the verdict and heuristics only nudge, verdicts should
stay stable under perturbations that defeat individual pixel heuristics. Writes robustness.json.

    python tests/adversarial_robustness.py            # confident images only (fast-ish)
    python tests/adversarial_robustness.py --all       # every labelled image
"""
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.image.preprocessor import load_image_from_path, load_image_from_base64
from core.image.models.cnn_detector import CNNDetector
from core.image.calibration import Calibrator
from config import CALIBRATION_PATH

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
AI_DIRS = ["tests/samples/ai", "tests/samples/ai_diverse", "tests/samples/ai_extra"]
REAL_DIRS = ["tests/samples/real_humans", "tests/samples/user_humans"]
PRIMARY_TRUST = 0.7


def _b64_jpeg(pil, quality=90):
    buf = io.BytesIO(); pil.convert("RGB").save(buf, "JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def perturb(pil, kind):
    if kind == "jpeg_q60":
        return pil  # applied via save quality below
    if kind == "gaussian_noise":
        arr = np.asarray(pil.convert("RGB")).astype(np.float32)
        arr += np.random.normal(0, 8.0, arr.shape)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if kind == "rescale_50":
        w, h = pil.size
        return pil.resize((max(1, w // 2), max(1, h // 2))).resize((w, h))
    if kind == "blur":
        return pil.filter(ImageFilter.GaussianBlur(radius=1.0))
    return pil


PERTURBATIONS = ["jpeg_q60", "gaussian_noise", "rescale_50", "blur"]


def verdict_of(det, cal, b64):
    r = det.analyze(load_image_from_base64(b64))
    if r.confidence < PRIMARY_TRUST or r.error:
        return "Inconclusive"
    return "AI Generated" if cal.apply(r.signal_score) > 0.5 else "Human Created"


def main():
    use_all = "--all" in sys.argv
    print("Loading dual-CNN…")
    det = CNNDetector(); cal = Calibrator.load(CALIBRATION_PATH)

    files = []
    for dirs, lab in ((AI_DIRS, 1), (REAL_DIRS, 0)):
        for d in dirs:
            p = Path(d)
            if p.is_dir():
                files += [(f, lab) for f in sorted(p.iterdir()) if f.suffix.lower() in IMAGE_EXT]

    results = {k: {"tested": 0, "flipped": 0} for k in PERTURBATIONS}
    checked = 0
    for f, lab in files:
        try:
            pil = Image.open(str(f)).convert("RGB")
            base_v = verdict_of(det, cal, _b64_jpeg(pil, 95))
        except Exception as e:  # noqa: BLE001
            print(f"  skip {f.name}: {e}"); continue
        # only probe images the model confidently decided (unless --all)
        if base_v == "Inconclusive" and not use_all:
            continue
        checked += 1
        for k in PERTURBATIONS:
            q = 60 if k == "jpeg_q60" else 95
            pv = verdict_of(det, cal, _b64_jpeg(perturb(pil, k), q))
            results[k]["tested"] += 1
            if pv != base_v:
                results[k]["flipped"] += 1
        print(f"  {f.name:38s} base={base_v}")

    for k in results:
        t = results[k]["tested"]
        results[k]["flip_rate"] = round(results[k]["flipped"] / t, 4) if t else None
    overall_t = sum(r["tested"] for r in results.values())
    overall_f = sum(r["flipped"] for r in results.values())
    out = {
        "images_probed": checked,
        "overall_flip_rate": round(overall_f / overall_t, 4) if overall_t else None,
        "per_perturbation": results,
        "note": "Flip = calibrated verdict changed vs the clean image. Lower is more robust. "
                "Perturbations: JPEG Q60, Gaussian noise, 50% rescale round-trip, mild blur.",
    }
    Path("robustness.json").write_text(json.dumps(out, indent=2))
    print("\n" + json.dumps(out, indent=2))
    print("\nwrote robustness.json")


if __name__ == "__main__":
    main()
