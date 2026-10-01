"""
Confidence calibration for the image detector.

Maps the primary (dual-CNN) raw AI score -> a calibrated P(AI) using Platt scaling
(a 2-parameter logistic fit by minimizing negative log-likelihood). Platt is chosen
deliberately over isotonic regression: our labeled validation set is tiny (~60 images),
and isotonic would overfit it. scikit-learn is not installed; this uses scipy only.

Honesty contract: if no calibration.json exists, `Calibrator.load` returns an IDENTITY
mapping with `calibrated=False`, and the judge labels its confidence "uncalibrated".
We never fabricate a calibrated number we didn't fit.
"""
import json
import logging
import os
from datetime import date
from typing import List, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def fit_platt(scores: Sequence[float], labels: Sequence[int]) -> Tuple[float, float]:
    """Fit P(AI) = sigmoid(A*score + B) by minimizing NLL with Platt target smoothing.

    labels: 1 = AI-generated, 0 = human. Returns (A, B).
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=float)
    n_pos = float(y.sum())
    n_neg = float(len(y) - n_pos)
    # Platt's smoothed targets guard against overconfidence on small samples.
    t = np.where(y == 1.0, (n_pos + 1.0) / (n_pos + 2.0), 1.0 / (n_neg + 2.0))

    def nll(params: np.ndarray) -> float:
        a, b = params
        p = np.clip(_sigmoid(a * s + b), 1e-9, 1 - 1e-9)
        return float(-np.sum(t * np.log(p) + (1 - t) * np.log(1 - p)))

    res = minimize(nll, x0=np.array([1.0, 0.0]), method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 2000})
    a, b = float(res.x[0]), float(res.x[1])
    return a, b


def expected_calibration_error(probs: Sequence[float], labels: Sequence[int], n_bins: int = 10) -> float:
    """Standard ECE over equal-width bins."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p > lo) & (p <= hi) if i > 0 else (p >= lo) & (p <= hi)
        if not mask.any():
            continue
        conf = p[mask].mean()
        acc = y[mask].mean()
        ece += (mask.mean()) * abs(acc - conf)
    return float(ece)


def leave_one_out_accuracy(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Honest small-sample estimate: refit on all-but-one, predict the held-out point."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=float)
    n = len(s)
    if n < 3:
        return float("nan")
    correct = 0
    for i in range(n):
        idx = [j for j in range(n) if j != i]
        a, b = fit_platt(s[idx], y[idx])
        p = _sigmoid(a * s[i] + b)
        correct += int((p > 0.5) == (y[i] > 0.5))
    return correct / n


class Calibrator:
    """Applies a fitted Platt mapping, or an identity map when uncalibrated."""

    def __init__(self, a: float = 1.0, b: float = 0.0, calibrated: bool = False, meta: dict | None = None):
        self.a = a
        self.b = b
        self.calibrated = calibrated
        self.meta = meta or {}

    def apply(self, score: float) -> float:
        if not self.calibrated:
            return float(np.clip(score, 0.0, 1.0))  # identity fallback
        return float(_sigmoid(self.a * float(score) + self.b))

    @classmethod
    def load(cls, path: str) -> "Calibrator":
        if not path or not os.path.exists(path):
            logger.info("No calibration file at %s — using identity map (uncalibrated).", path)
            return cls(calibrated=False)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                d = json.load(fh)
            return cls(a=float(d["A"]), b=float(d["B"]), calibrated=True, meta=d)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to load calibration %s: %s — using identity map.", path, e)
            return cls(calibrated=False)

    def save(self, path: str, scores: Sequence[float], labels: Sequence[int]) -> dict:
        """Fit on (scores, labels), persist, and return the metrics dict."""
        a, b = fit_platt(scores, labels)
        self.a, self.b, self.calibrated = a, b, True
        cal_probs = [self.apply(x) for x in scores]
        meta = {
            "A": a,
            "B": b,
            "n_samples": int(len(scores)),
            "n_ai": int(np.sum(np.asarray(labels) == 1)),
            "n_human": int(np.sum(np.asarray(labels) == 0)),
            "loo_accuracy": round(leave_one_out_accuracy(scores, labels), 4),
            "ece": round(expected_calibration_error(cal_probs, labels), 4),
            "method": "platt-2param",
            "fitted_on": date.today().isoformat(),
        }
        self.meta = meta
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        logger.info("Saved calibration to %s: %s", path, meta)
        return meta


if __name__ == "__main__":
    # ponytail: self-check — a separable toy set must calibrate and classify correctly.
    rng = np.random.default_rng(0)
    ai = rng.uniform(0.55, 1.0, 40)
    hum = rng.uniform(0.0, 0.45, 40)
    xs = np.concatenate([ai, hum])
    ys = np.concatenate([np.ones(40), np.zeros(40)]).astype(int)
    A, B = fit_platt(xs, ys)
    c = Calibrator(A, B, calibrated=True)
    assert c.apply(0.9) > 0.5, "high score must map to AI"
    assert c.apply(0.1) < 0.5, "low score must map to human"
    loo = leave_one_out_accuracy(xs, ys)
    assert loo > 0.9, f"separable data should give high LOO acc, got {loo}"
    # identity fallback
    ident = Calibrator(calibrated=False)
    assert ident.apply(0.73) == 0.73
    print(f"calibration self-check OK (A={A:.3f} B={B:.3f} LOO={loo:.3f})")
