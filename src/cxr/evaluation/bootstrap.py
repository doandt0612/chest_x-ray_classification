"""Bootstrap confidence intervals for AUC, sensitivity and specificity at a fixed threshold.

Resampling is stratified by class: each round draws the NORMAL and the PNEUMONIA images
with replacement separately, so every resample keeps the original class counts. This keeps
AUC defined in every round (both classes always present) and matches how the test set is
fixed: its class balance is a property of the data, not of chance.

CI = percentile interval of the resampled values. The threshold is not re-chosen per round:
it was picked on val and stays fixed, exactly as it will be in deployment.

Caveat: when a metric is exactly 0 or 1 on the full data (e.g. no NORMAL image misclassified),
every resample gives the same value and the interval has zero width. That means "not
estimable by bootstrap", not "no uncertainty"; report it as such.
"""

import numpy as np

from cxr.evaluation.metrics import binary_metrics

CI_METRICS = ("auc", "sensitivity", "specificity")


def bootstrap_ci(y_true, prob, threshold: float, cfg: dict) -> dict:
    """Point estimate + percentile CI for each metric in CI_METRICS.

    cfg: the `bootstrap` section of configs/eval.yaml (n_boot, confidence, seed).
    Returns {"auc": {"value", "ci_low", "ci_high"}, ..., "n_boot", "confidence", "seed"}.
    """
    y, prob = np.asarray(y_true), np.asarray(prob)
    pos_idx, neg_idx = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError("Bootstrap needs both classes in y_true")
    n_boot, confidence = cfg["n_boot"], cfg["confidence"]
    if n_boot < 1 or not 0 < confidence < 1:
        raise ValueError(f"Need n_boot >= 1 and 0 < confidence < 1, got {n_boot}, {confidence}")

    rng = np.random.default_rng(cfg["seed"])
    samples = np.empty((n_boot, len(CI_METRICS)))
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(pos_idx, len(pos_idx)), rng.choice(neg_idx, len(neg_idx))])
        m = binary_metrics(y[idx], prob[idx], threshold)
        samples[b] = [m[name] for name in CI_METRICS]

    point = binary_metrics(y, prob, threshold)
    tail = (1 - confidence) / 2 * 100
    low, high = np.percentile(samples, [tail, 100 - tail], axis=0)
    result = {
        name: {"value": point[name], "ci_low": float(low[i]), "ci_high": float(high[i])}
        for i, name in enumerate(CI_METRICS)
    }
    return {**result, "n_boot": n_boot, "confidence": confidence, "seed": cfg["seed"]}
