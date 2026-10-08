"""
Validation metrics for the image primary detector: ROC/AUC, confusion matrix, per-class
accuracy, and the escalation rate. Writes evaluation.json for honest reporting.

    python tests/evaluate.py

Uses the labelled sample set. scikit-learn (installed transitively via librosa) provides
the metrics; falls back to a hand-rolled AUC if it is somehow absent.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.image.preprocessor import load_image_from_path
from core.image.models.cnn_detector import CNNDetector
from core.image.calibration import Calibrator
from config import CALIBRATION_PATH

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
AI_DIRS = ["tests/samples/ai", "tests/samples/ai_diverse", "tests/samples/ai_extra"]
REAL_DIRS = ["tests/samples/real_humans", "tests/samples/user_humans"]
PRIMARY_TRUST = 0.7


def _auc(scores, labels):
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(labels, scores))
    except Exception:
        # Mann-Whitney U fallback
        pos = [s for s, y in zip(scores, labels) if y == 1]
        neg = [s for s, y in zip(scores, labels) if y == 0]
        if not pos or not neg:
            return float("nan")
        wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
        return wins / (len(pos) * len(neg))


def collect(det, dirs, label):
    rows = []
    for d in dirs:
        p = Path(d)
        if not p.is_dir():
            continue
        for f in sorted(p.iterdir()):
            if f.suffix.lower() not in IMAGE_EXT:
                continue
            try:
                r = det.analyze(load_image_from_path(str(f)))
            except Exception as e:  # noqa: BLE001
                print(f"  skip {f.name}: {e}")
                continue
            rows.append({"name": f.name, "label": label, "score": float(r.signal_score),
                         "conf": float(r.confidence), "confident": r.confidence >= PRIMARY_TRUST and not r.error})
    return rows


def main():
    print("Loading dual-CNN (downloads once)…")
    det = CNNDetector()
    rows = collect(det, AI_DIRS, 1) + collect(det, REAL_DIRS, 0)
    if len(rows) < 6:
        print("Not enough samples."); sys.exit(1)

    cal = Calibrator.load(CALIBRATION_PATH)
    scores = [r["score"] for r in rows]
    labels = [r["label"] for r in rows]

    auc_all = _auc(scores, labels)
    conf_rows = [r for r in rows if r["confident"]]
    escalation_rate = 1 - len(conf_rows) / len(rows)

    # Operating point: calibrated prob > 0.5 on the confident population (what users see decided).
    tp = fp = tn = fn = 0
    for r in conf_rows:
        pred_ai = cal.apply(r["score"]) > 0.5
        if r["label"] == 1 and pred_ai: tp += 1
        elif r["label"] == 1 and not pred_ai: fn += 1
        elif r["label"] == 0 and not pred_ai: tn += 1
        else: fp += 1
    acc = (tp + tn) / len(conf_rows) if conf_rows else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    auc_conf = _auc([r["score"] for r in conf_rows], [r["label"] for r in conf_rows])

    out = {
        "n_total": len(rows), "n_ai": sum(labels), "n_human": len(labels) - sum(labels),
        "auc_all": round(auc_all, 4),
        "escalation_rate": round(escalation_rate, 4),
        "n_confident": len(conf_rows),
        "confident": {
            "auc": round(auc_conf, 4), "accuracy": round(acc, 4),
            "precision_ai": round(prec, 4), "recall_ai": round(rec, 4),
            "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        },
        "calibrated": cal.calibrated,
        "note": "Operating point = calibrated P(AI)>0.5 on the confident-primary population; "
                "the rest escalate to human review.",
    }
    Path("evaluation.json").write_text(json.dumps(out, indent=2))
    print("\n" + json.dumps(out, indent=2))
    print("\nwrote evaluation.json")


if __name__ == "__main__":
    main()
