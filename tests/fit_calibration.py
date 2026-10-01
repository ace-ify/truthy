"""
Fit the image-detector confidence calibration (Platt scaling) on the labeled sample set,
and report honest metrics (leave-one-out accuracy, ECE). Writes calibration.json.

    python tests/fit_calibration.py

Only the PRIMARY signal (dual-CNN) is calibrated — that is the number the judge trusts.
Downloads the two HF models on first run.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.image.preprocessor import load_image_from_path
from core.image.models.cnn_detector import CNNDetector
from core.image.calibration import Calibrator, leave_one_out_accuracy, expected_calibration_error
from config import CALIBRATION_PATH

# We calibrate the number the judge actually USES: the primary CNN score on samples where
# the CNN was confident enough to commit (>= this). Neutral "disagreement" outputs (~0.5)
# become Inconclusive+escalate in the judge and never use calibration, so including them
# would only flatten the fit and make real detections under-confident.
PRIMARY_TRUST_CONFIDENCE = 0.7

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
AI_DIRS = ["tests/samples/ai", "tests/samples/ai_diverse", "tests/samples/ai_extra"]
REAL_DIRS = ["tests/samples/real_humans", "tests/samples/user_humans"]


def collect(det, dirs, label):
    """Return (confident_scores, all_count, escalated_count) for a label group."""
    scores, total, escalated = [], 0, 0
    for d in dirs:
        p = Path(d)
        if not p.is_dir():
            continue
        for f in sorted(p.iterdir()):
            if f.suffix.lower() not in IMAGE_EXT:
                continue
            total += 1
            try:
                img = load_image_from_path(str(f))
                r = det.analyze(img)
            except Exception as e:  # noqa: BLE001
                print(f"  skip {f.name}: {e}")
                continue
            if r.confidence < PRIMARY_TRUST_CONFIDENCE or r.error:
                escalated += 1
                print(f"  {f.name:40s} label={label} cnn={r.signal_score:.3f} "
                      f"({r.details.get('method')}) -> escalate (low conf)")
                continue
            scores.append(float(r.signal_score))
            print(f"  {f.name:40s} label={label} cnn={r.signal_score:.3f} ({r.details.get('method')})")
    return scores, total, escalated


def main():
    print("Loading dual-CNN detector (downloads models on first run)...")
    det = CNNDetector()

    print("\n== AI samples ==")
    ai_scores, ai_total, ai_esc = collect(det, AI_DIRS, 1)
    print("\n== Real samples ==")
    real_scores, real_total, real_esc = collect(det, REAL_DIRS, 0)

    scores = ai_scores + real_scores
    labels = [1] * len(ai_scores) + [0] * len(real_scores)
    total_imgs = ai_total + real_total
    escalated = ai_esc + real_esc
    if len(scores) < 6:
        print(f"\nNot enough confident samples ({len(scores)}). Aborting.")
        sys.exit(1)

    # Raw accuracy at the 0.5 threshold on the raw CNN score (baseline).
    raw_correct = sum(1 for s, y in zip(scores, labels) if (s > 0.5) == (y == 1))
    raw_acc = raw_correct / len(scores)

    cal = Calibrator()
    meta = cal.save(CALIBRATION_PATH, scores, labels)

    cal_probs = [cal.apply(s) for s in scores]
    cal_correct = sum(1 for p, y in zip(cal_probs, labels) if (p > 0.5) == (y == 1))
    cal_acc = cal_correct / len(scores)

    print("\n" + "=" * 56)
    print("  CALIBRATION FITTED (confident-primary population)")
    print("=" * 56)
    print(f"  images seen:             {total_imgs}")
    print(f"  routed to review (low-conf CNN): {escalated} ({escalated/total_imgs:.0%})")
    print(f"  confident samples fit:   {meta['n_samples']}  (AI={meta['n_ai']}, human={meta['n_human']})")
    print(f"  Platt: P(AI) = sigmoid({meta['A']:.3f}*score + {meta['B']:.3f})")
    print(f"  accuracy on confident (raw):        {raw_acc:.1%}")
    print(f"  accuracy on confident (calibrated): {cal_acc:.1%}")
    print(f"  leave-one-out accuracy:             {meta['loo_accuracy']:.1%}")
    print(f"  expected calibration error:         {meta['ece']:.3f}")
    print(f"  written to: {CALIBRATION_PATH}")
    print("=" * 56)


if __name__ == "__main__":
    main()
