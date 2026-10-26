import numpy as np
import pytest

from cxr.evaluation.bootstrap import bootstrap_ci
from cxr.evaluation.metrics import binary_metrics

CFG = {"n_boot": 200, "confidence": 0.95, "seed": 0}  # small n_boot keeps tests fast


def test_perfect_separation_gives_degenerate_intervals():
    # Every resample is still perfectly separated, so all three metrics are exactly 1.
    y = np.array([0] * 20 + [1] * 30)
    prob = np.where(y == 1, 0.9, 0.1)
    r = bootstrap_ci(y, prob, 0.5, CFG)
    for name in ("auc", "sensitivity", "specificity"):
        assert r[name] == {"value": 1.0, "ci_low": 1.0, "ci_high": 1.0}


def test_sensitivity_ci_roughly_matches_binomial_width():
    # 400 positives, 360 caught -> sensitivity 0.9; normal approximation of a 95% CI:
    # 0.9 +/- 1.96 * sqrt(0.9 * 0.1 / 400) = 0.9 +/- 0.0294. Sanity check, not an exact match.
    y = np.array([1] * 400 + [0] * 200)
    prob = np.concatenate([np.full(360, 0.8), np.full(40, 0.2), np.full(200, 0.1)])
    r = bootstrap_ci(y, prob, 0.5, {**CFG, "n_boot": 1000})
    assert r["sensitivity"]["value"] == 0.9
    assert r["sensitivity"]["ci_low"] == pytest.approx(0.9 - 0.0294, abs=0.01)
    assert r["sensitivity"]["ci_high"] == pytest.approx(0.9 + 0.0294, abs=0.01)
    # Every NORMAL scores below the threshold, in every resample.
    assert r["specificity"] == {"value": 1.0, "ci_low": 1.0, "ci_high": 1.0}


def test_point_values_match_binary_metrics_and_lie_inside_ci():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    prob = np.clip(0.3 * y + rng.normal(0.35, 0.2, 300), 0, 1)
    r = bootstrap_ci(y, prob, 0.5, CFG)
    point = binary_metrics(y, prob, 0.5)
    for name in ("auc", "sensitivity", "specificity"):
        assert r[name]["value"] == point[name]
        assert r[name]["ci_low"] < r[name]["ci_high"]
    assert (r["n_boot"], r["confidence"], r["seed"]) == (200, 0.95, 0)


def test_same_seed_same_result_different_seed_differs():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 200)
    prob = rng.random(200)
    assert bootstrap_ci(y, prob, 0.5, CFG) == bootstrap_ci(y, prob, 0.5, CFG)
    other = bootstrap_ci(y, prob, 0.5, {**CFG, "seed": 1})
    assert other["auc"]["ci_low"] != bootstrap_ci(y, prob, 0.5, CFG)["auc"]["ci_low"]


def test_confidence_is_used():
    # Same seed -> same resamples, so this only checks that `confidence` reaches the percentiles.
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 200)
    prob = np.clip(0.3 * y + rng.normal(0.35, 0.2, 200), 0, 1)
    narrow = bootstrap_ci(y, prob, 0.5, {**CFG, "confidence": 0.80})["auc"]
    wide = bootstrap_ci(y, prob, 0.5, CFG)["auc"]
    assert wide["ci_low"] < narrow["ci_low"] and narrow["ci_high"] < wide["ci_high"]


def test_resampling_is_stratified():
    # 1 positive in 50: an unstratified resample misses it ~36% of the time and AUC is undefined.
    # Stratified, the positive is always drawn, so sensitivity stays exactly 1 and AUC is defined.
    y = np.array([1] + [0] * 49)
    prob = np.random.default_rng(4).random(50) * 0.4
    prob[0] = 0.9
    r = bootstrap_ci(y, prob, 0.5, CFG)
    assert r["sensitivity"] == {"value": 1.0, "ci_low": 1.0, "ci_high": 1.0}
    assert r["auc"]["ci_low"] == pytest.approx(1.0)  # percentile interpolation leaves ~1e-16


@pytest.mark.parametrize(
    "y, cfg",
    [
        ([1, 1, 1], CFG),  # one class only
        ([0, 1, 1], {**CFG, "n_boot": 0}),
        ([0, 1, 1], {**CFG, "confidence": 1.0}),
    ],
)
def test_invalid_input_raises(y, cfg):
    with pytest.raises(ValueError):
        bootstrap_ci(y, [0.2, 0.6, 0.9], 0.5, cfg)
