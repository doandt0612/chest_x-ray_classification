import pytest

from cxr.monitoring.drift import check_quality, check_reference_matches, load_bounds

REFERENCE = {
    "image": {"size": 224, "resize": "squash"},
    "stats": {
        "brightness": {"quantiles": {"p0.5": 0.2, "p99.5": 0.7}},
        "noise": {"quantiles": {"p0.5": 0.002, "p99.5": 0.006}},
    },
}
CHECKS = {"brightness": ["p0.5", "p99.5"], "noise": [None, "p99.5"]}


def test_load_bounds_reads_quantiles_and_keeps_unchecked_side():
    assert load_bounds(REFERENCE, CHECKS) == {"brightness": (0.2, 0.7), "noise": (None, 0.006)}


def test_inside_bounds_is_ok_and_bounds_are_inclusive():
    bounds = load_bounds(REFERENCE, CHECKS)
    assert check_quality({"brightness": 0.5, "noise": 0.004}, bounds) == {"status": "ok", "flags": []}
    assert check_quality({"brightness": 0.2, "noise": 0.006}, bounds)["status"] == "ok"


def test_out_of_bounds_flags_side_and_limit():
    bounds = load_bounds(REFERENCE, CHECKS)
    result = check_quality({"brightness": 0.1, "noise": 0.01}, bounds)
    assert result["status"] == "warning"
    assert result["flags"] == [
        {"stat": "brightness", "value": 0.1, "side": "low", "limit": 0.2},
        {"stat": "noise", "value": 0.01, "side": "high", "limit": 0.006},
    ]


def test_unchecked_side_never_flags():
    bounds = load_bounds(REFERENCE, CHECKS)
    # noise has no low bound: a very clean image is fine
    assert check_quality({"brightness": 0.5, "noise": 0.0}, bounds)["status"] == "ok"


def test_unknown_quantile_key_raises():
    with pytest.raises(KeyError):
        load_bounds(REFERENCE, {"brightness": ["p0.1", "p99.5"]})


def test_reference_must_match_model_preprocessing():
    check_reference_matches(REFERENCE, {"size": 224, "resize": "squash", "mean": [0.5] * 3})
    with pytest.raises(ValueError, match="resize"):
        check_reference_matches(REFERENCE, {"size": 224, "resize": "center_crop"})
    with pytest.raises(ValueError, match="size"):
        check_reference_matches(REFERENCE, {"size": 256, "resize": "squash"})
