import pandas as pd
import pytest

from cxr.monitoring.evaluate_monitoring import check_acceptance, summarize, variants

CFG = {
    "checks": {"brightness": ["p0.5", "p99.5"], "blur": ["p0.5", None]},
    "degradations": {"blur": [2, 8], "dark": [0.5]},
    "acceptance": {"max_false_alarm": 0.05, "min_detection": 0.9},
}


def _rows(kind, level, labels, probs, flags):
    return [
        {
            "image_id": f"i{i}",
            "label": y,
            "kind": kind,
            "level": level,
            "prob": p,
            "warning": bool(f),
            "flags": f,
        }
        for i, (y, p, f) in enumerate(zip(labels, probs, flags))
    ]


LABELS = [0, 0, 1, 1]
RESULTS = pd.DataFrame(
    _rows("clean", None, LABELS, [0.1, 0.2, 0.8, 0.9], ["", "", "", "brightness:high"])
    + _rows("blur", 2, LABELS, [0.1, 0.6, 0.4, 0.9], ["", "blur:low", "", ""])
    + _rows("blur", 8, LABELS, [0.5, 0.6, 0.4, 0.3], ["blur:low"] * 4)
    + _rows("dark", 0.5, LABELS, [0.1, 0.2, 0.8, 0.9], ["brightness:low", "brightness:low;blur:low", "", ""])
)


def test_variants_start_with_clean_in_config_order():
    assert variants(CFG["degradations"]) == [("clean", None), ("blur", 2), ("blur", 8), ("dark", 0.5)]
    with pytest.raises(ValueError, match="Unknown degradations"):
        variants({"rotate": [1]})


def test_summarize_rates_and_metrics():
    table = summarize(RESULTS, CFG, threshold=0.5).set_index(["kind", "level"], drop=False)
    clean = table.iloc[0]
    assert clean["n"] == 4
    assert clean["warning_rate"] == 0.25
    assert clean["flag_brightness"] == 0.25 and clean["flag_blur"] == 0.0
    assert clean["auc"] == 1.0 and clean["sensitivity"] == 1.0 and clean["specificity"] == 1.0

    blur2 = table[(table["kind"] == "blur") & (table["level"] == 2)].iloc[0]
    assert blur2["warning_rate"] == 0.25 and blur2["flag_blur"] == 0.25
    # probs 0.1, 0.6 (NORMAL) vs 0.4, 0.9 (PNEUMONIA): 3 of 4 pairs ranked right
    assert blur2["auc"] == 0.75
    assert blur2["sensitivity"] == 0.5 and blur2["specificity"] == 0.5

    dark = table[table["kind"] == "dark"].iloc[0]
    assert dark["warning_rate"] == 0.5
    assert dark["flag_brightness"] == 0.5 and dark["flag_blur"] == 0.25


def test_acceptance_uses_clean_and_strongest_level():
    acc = check_acceptance(summarize(RESULTS, CFG, threshold=0.5), CFG)
    assert acc["false_alarm"] == {"max": 0.05, "value": 0.25, "passed": False}
    assert acc["detection_blur"] == {"level": 8, "min": 0.9, "value": 1.0, "passed": True}  # level 2 ignored
    assert acc["detection_dark"]["value"] == 0.5 and not acc["detection_dark"]["passed"]
