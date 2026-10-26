import math

import numpy as np
import pytest

from cxr.evaluation.metrics import binary_metrics, choose_threshold

# 4 NORMAL (0) then 4 PNEUMONIA (1); one NORMAL scores above one PNEUMONIA.
Y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
P = np.array([0.1, 0.2, 0.3, 0.7, 0.6, 0.8, 0.9, 0.95])


def test_binary_metrics_counts_and_rates():
    m = binary_metrics(Y, P, threshold=0.5)
    # prob >= 0.5 -> positive: NORMAL 0.7 is a false positive, every PNEUMONIA is caught.
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (3, 1, 0, 4)
    assert m["sensitivity"] == 1.0 and m["specificity"] == 0.75
    assert m["auc"] == pytest.approx(15 / 16)  # 15 of the 16 (pos, neg) pairs are ranked correctly
    assert m["n"] == 8 and m["threshold"] == 0.5


def test_threshold_is_inclusive():
    m = binary_metrics(Y, P, threshold=0.7)  # 0.7 itself counts as positive
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (3, 1, 1, 3)


@pytest.mark.filterwarnings("ignore:Only one class")
def test_single_class_gives_nan_instead_of_crashing():
    m = binary_metrics([1, 1], [0.9, 0.3], threshold=0.5)
    assert (m["tp"], m["fn"]) == (1, 1) and m["sensitivity"] == 0.5
    assert math.isnan(m["specificity"]) and math.isnan(m["auc"])  # no NORMAL images


def test_youden_threshold():
    # Thresholds 0.6 and 0.8 both give J = 0.75; the higher one is chosen on a tie.
    t = choose_threshold(Y, P, {"method": "youden"})
    assert t == 0.8
    m = binary_metrics(Y, P, t)
    assert (m["sensitivity"], m["specificity"]) == (0.75, 1.0)


def test_min_sensitivity_threshold():
    t = choose_threshold(Y, P, {"method": "min_sensitivity", "target_sensitivity": 1.0})
    assert t == 0.6  # lowest PNEUMONIA score: catches all 4, lets one NORMAL through
    t = choose_threshold(Y, P, {"method": "min_sensitivity", "target_sensitivity": 0.5})
    assert t == 0.9  # 2 of 4 caught, no false positive


def test_invalid_config_raises():
    with pytest.raises(ValueError, match="Unknown threshold method"):
        choose_threshold(Y, P, {"method": "f1"})
    with pytest.raises(ValueError, match="target_sensitivity"):
        choose_threshold(Y, P, {"method": "min_sensitivity", "target_sensitivity": 95})
