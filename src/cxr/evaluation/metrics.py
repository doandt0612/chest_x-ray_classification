"""Binary classification metrics and decision threshold selection.

Positive class = 1 = PNEUMONIA. An image is predicted positive when prob >= threshold.
The threshold is chosen on val only; test metrics reuse that threshold unchanged.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

THRESHOLD_METHODS = ("youden", "min_sensitivity")


def _ratio(num: int, den: int) -> float:
    return num / den if den else float("nan")


def binary_metrics(y_true, prob, threshold: float) -> dict:
    """AUC (threshold-free), then sensitivity, specificity and confusion counts at `threshold`."""
    y_true, prob = np.asarray(y_true), np.asarray(prob)
    pred = prob >= threshold
    tp = int(np.sum(pred & (y_true == 1)))
    fn = int(np.sum(~pred & (y_true == 1)))
    tn = int(np.sum(~pred & (y_true == 0)))
    fp = int(np.sum(pred & (y_true == 0)))
    return {
        "n": len(y_true),
        "auc": float(roc_auc_score(y_true, prob)),
        "threshold": float(threshold),
        "sensitivity": _ratio(tp, tp + fn),  # recall of PNEUMONIA
        "specificity": _ratio(tn, tn + fp),  # recall of NORMAL
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def choose_threshold(y_true, prob, cfg: dict) -> float:
    """Pick the decision threshold on val. cfg: the `threshold` section of configs/eval.yaml.

    youden: maximize sensitivity + specificity - 1.
    min_sensitivity: the highest threshold whose sensitivity is >= cfg["target_sensitivity"],
    i.e. the best specificity that still meets the sensitivity target.
    """
    method = cfg["method"]
    if method not in THRESHOLD_METHODS:
        raise ValueError(f"Unknown threshold method {method!r}; expected one of {THRESHOLD_METHODS}")

    # roc_curve: thresholds decrease, tpr/fpr increase; thresholds[0] is +inf (predict nothing).
    # drop_intermediate=False keeps every distinct score, so ties resolve to the highest threshold.
    fpr, tpr, thresholds = roc_curve(y_true, prob, drop_intermediate=False)
    fpr, tpr, thresholds = fpr[1:], tpr[1:], thresholds[1:]
    if method == "youden":
        idx = int(np.argmax(tpr - fpr))  # first maximum = highest threshold among ties
    else:
        target = cfg["target_sensitivity"]
        if not 0 < target <= 1:
            raise ValueError(f"target_sensitivity must be in (0, 1], got {target}")
        idx = int(np.argmax(tpr >= target))  # first index reaching the target
    return float(thresholds[idx])
