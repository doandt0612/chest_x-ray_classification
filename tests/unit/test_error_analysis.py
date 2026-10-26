import numpy as np
import pandas as pd
import pytest
from PIL import Image

from cxr.evaluation.error_analysis import (
    analyze,
    by_height_third,
    compare_stats,
    outcomes,
    prob_bands,
    size_label_auc,
)
from cxr.monitoring.image_stats import STAT_NAMES


def test_outcomes_follow_inclusive_threshold():
    label = pd.Series([0, 0, 1, 1, 0])
    prob = pd.Series([0.1, 0.3, 0.2, 0.9, 0.25])
    # 0.25 == threshold -> positive, so the last NORMAL is a false positive.
    assert outcomes(label, prob, 0.25).tolist() == ["TN", "FP", "FN", "TP", "FP"]


def test_prob_bands_split_false_positives():
    prob = pd.Series([0.25, 0.49, 0.5, 0.89, 0.9, 1.0])
    assert prob_bands(prob, 0.25) == {"[0.25, 0.5)": 2, "[0.5, 0.9)": 2, "[0.9, 1]": 2}


def test_prob_bands_rejects_threshold_above_first_band():
    with pytest.raises(ValueError, match="threshold"):
        prob_bands(pd.Series([0.7]), 0.6)


def _stats(value: float, n: int) -> pd.DataFrame:
    return pd.DataFrame({name: [value + i for i in range(n)] for name in STAT_NAMES})


def test_compare_stats_auc_and_medians():
    a, b = _stats(100, 3), _stats(0, 4)  # every `a` value is larger than every `b` value
    r = compare_stats(a, b)
    assert set(r) == set(STAT_NAMES)
    assert r["width"] == {"median_a": 101.0, "median_b": 1.5, "auc": 1.0}
    assert compare_stats(b, a)["width"]["auc"] == 0.0
    assert compare_stats(a, a)["width"]["auc"] == 0.5  # identical groups: no separation


def test_size_label_auc_per_split():
    # In "x", PNEUMONIA images are shorter and wider; in "y", size carries no information.
    sizes = pd.DataFrame(
        {
            "split": ["x"] * 4 + ["y"] * 4,
            "label": [0, 0, 1, 1, 0, 1, 0, 1],
            "width": [100, 100, 120, 120, 100, 100, 100, 100],
            "height": [100, 100, 80, 80, 100, 100, 100, 100],
        }
    )
    r = size_label_auc(sizes)
    assert r["x"] == {"height": 0.0, "aspect_ratio": 1.0}
    assert r["y"] == {"height": pytest.approx(0.5), "aspect_ratio": pytest.approx(0.5)}


def test_by_height_third_fp_rate_and_ranges():
    # 6 NORMAL images, heights 100..600; the two smallest are false positives.
    stats = pd.DataFrame({"height": [100.0, 200, 300, 400, 500, 600], "aspect_ratio": [1.0] * 6})
    normal = pd.DataFrame(
        {"outcome": ["FP", "FP", "TN", "TN", "TN", "TN"], "prob": [0.9, 0.8, 0.1, 0.1, 0.1, 0.1]}
    )
    rows = by_height_third(normal, stats)
    assert [(t["third"], t["n"], t["height_min"], t["height_max"], t["fp_rate"]) for t in rows] == [
        ("small", 2, 100.0, 200.0, 1.0),
        ("middle", 2, 300.0, 400.0, 0.0),
        ("large", 2, 500.0, 600.0, 0.0),
    ]


def test_analyze_on_tiny_images(tmp_path):
    # Threshold 0.3. Test: 6 NORMAL (heights 40..90, the 2 smallest are FPs), 2 PNEUMONIA (1 missed).
    def image(name: str, height: int) -> str:
        path = tmp_path / f"{name}.png"
        Image.fromarray(np.full((height, 60), 128, dtype=np.uint8)).save(path)
        return str(path)

    heights = [40, 50, 60, 70, 80, 90]
    test = pd.DataFrame(
        {
            "image_id": [f"t{i}" for i in range(8)],
            "label": [0] * 6 + [1, 1],
            "prob": [0.9, 0.6, 0.2, 0.1, 0.05, 0.01, 0.95, 0.2],
            "path": [image(f"t{i}", h) for i, h in enumerate(heights + [40, 40])],
            "patient_id": ["p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7"],
        }
    )
    val = pd.DataFrame(
        {
            "image_id": ["v0", "v1", "v2", "v3"],
            "label": [0, 0, 0, 1],
            "prob": [0.01, 0.02, 0.6, 0.9],
            "path": [image(f"v{i}", 80) for i in range(4)],
            "patient_id": ["q0", "q1", "q2", "q3"],
        }
    )
    sizes = pd.DataFrame(
        {"split": ["test"] * 4, "label": [0, 0, 1, 1], "width": [60] * 4, "height": [90, 80, 40, 50]}
    )
    image_cfg = {"size": 32, "resize": "squash", "mean": [0.5] * 3, "std": [0.5] * 3}
    r = analyze(test, val, sizes, 0.3, image_cfg)

    assert r["test_outcomes"] == {"TN": 4, "FP": 2, "FN": 1, "TP": 1}
    assert r["normal_above_threshold"] == {"val": pytest.approx(1 / 3), "test": pytest.approx(2 / 6)}
    assert r["fp_patients"] == 2 and r["tn_patients"] == 4
    assert r["normal_prob_spearman"]["height"] == pytest.approx(-1.0)  # prob falls as height rises
    assert [t["fp_rate"] for t in r["normal_by_height_third"]] == [1.0, 0.0, 0.0]
    assert r["fp_prob_bands"] == {"[0.3, 0.5)": 0, "[0.5, 0.9)": 1, "[0.9, 1]": 1}
    assert [d["image_id"] for d in r["false_positives"]] == ["t1", "t0"]  # sorted by prob
    assert r["fp_vs_tn"]["height"]["auc"] == 0.0  # every FP is shorter than every TN
