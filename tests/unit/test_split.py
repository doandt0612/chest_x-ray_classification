import pandas as pd
import pytest

from cxr.data.split import check_no_group_leakage, make_splits


def _cfg(**overrides):
    split = {
        "val_fraction": 0.2,
        "seed": 0,
        "merge_hf_validation": True,
        "drop_exact_duplicates": True,
        "exclude_flags": [],
    }
    split.update(overrides)
    return {"split": split}


def _data(n_groups=60, per_group=3):
    """Train pool of n_groups groups (alternating labels), plus 2 HF-validation and 10 test images."""
    rows = []
    for g in range(n_groups):
        label = g % 2
        for k in range(per_group):
            rows.append(("train", label, f"g{g:03d}", f"p{g}"))
    rows += [("validation", 0, "gv0", "pv0"), ("validation", 1, "gv1", "pv1")]
    rows += [("test", i % 2, f"gt{i}", f"pt{i}") for i in range(10)]

    dedup = pd.DataFrame(rows, columns=["split", "label", "group_id", "patient_id"])
    dedup["image_id"] = [f"img{i:04d}" for i in range(len(dedup))]
    dedup["label_name"] = dedup["label"].map({0: "NORMAL", 1: "PNEUMONIA"})
    dedup["path"] = dedup["image_id"] + ".jpeg"
    validation = dedup[["image_id"]].assign(too_small=False)
    pairs = pd.DataFrame(columns=["image_id_a", "image_id_b", "exact"]).astype({"exact": bool})
    return dedup, validation, pairs


def test_groups_never_span_splits():
    dedup, validation, pairs = _data()
    splits, _ = make_splits(dedup, validation, pairs, _cfg())
    assert (splits.groupby("group_id")["split"].nunique() == 1).all()


def test_test_set_is_kept_unchanged():
    dedup, validation, pairs = _data()
    splits, _ = make_splits(dedup, validation, pairs, _cfg())
    test_ids = set(dedup.loc[dedup["split"] == "test", "image_id"])
    assert set(splits.loc[splits["split"] == "test", "image_id"]) == test_ids


def test_val_is_stratified_and_sized():
    dedup, validation, pairs = _data()
    splits, _ = make_splits(dedup, validation, pairs, _cfg())
    val = splits[splits["split"] == "val"]
    pool = splits[splits["split"] != "test"]
    assert 0.15 <= len(val) / len(pool) <= 0.25
    assert abs(val["label"].mean() - pool["label"].mean()) < 0.1


def test_split_is_deterministic():
    dedup, validation, pairs = _data()
    a, _ = make_splits(dedup, validation, pairs, _cfg())
    b, _ = make_splits(dedup, validation, pairs, _cfg())
    pd.testing.assert_frame_equal(a, b)


def test_hf_validation_merge_flag():
    dedup, validation, pairs = _data()
    merged, _ = make_splits(dedup, validation, pairs, _cfg(merge_hf_validation=True))
    dropped, _ = make_splits(dedup, validation, pairs, _cfg(merge_hf_validation=False))
    assert (merged["source_split"] == "validation").sum() == 2
    assert (dropped["source_split"] == "validation").sum() == 0


def test_exact_duplicates_dropped_from_pool_only():
    dedup, validation, _ = _data()
    train_ids = dedup.loc[dedup["split"] == "train", "image_id"].tolist()
    test_ids = dedup.loc[dedup["split"] == "test", "image_id"].tolist()
    pairs = pd.DataFrame(
        {
            "image_id_a": [train_ids[0], test_ids[0]],
            "image_id_b": [train_ids[1], test_ids[1]],
            "exact": [True, True],
        }
    )
    splits, excluded = make_splits(dedup, validation, pairs, _cfg())
    assert excluded["exact_duplicate"] == 1
    assert train_ids[0] in set(splits["image_id"]) and train_ids[1] not in set(splits["image_id"])
    assert {test_ids[0], test_ids[1]} <= set(splits["image_id"])


def test_exclude_flags_drop_from_pool_only():
    dedup, validation, pairs = _data()
    flagged = [dedup.loc[dedup["split"] == "train", "image_id"].iloc[0],
               dedup.loc[dedup["split"] == "test", "image_id"].iloc[0]]
    validation.loc[validation["image_id"].isin(flagged), "too_small"] = True
    splits, excluded = make_splits(dedup, validation, pairs, _cfg(exclude_flags=["too_small"]))
    assert excluded == {"too_small": 1, "flagged_total": 1, "exact_duplicate": 0}
    assert flagged[0] not in set(splits["image_id"]) and flagged[1] in set(splits["image_id"])


def test_overlapping_flags_counted_once_in_total():
    dedup, validation, pairs = _data()
    train_ids = dedup.loc[dedup["split"] == "train", "image_id"].tolist()
    validation["rgb_mode"] = validation["image_id"].isin(train_ids[:3])
    validation["too_small"] = validation["image_id"].isin(train_ids[2:4])
    _, excluded = make_splits(dedup, validation, pairs, _cfg(exclude_flags=["too_small", "rgb_mode"]))
    assert excluded["too_small"] == 2 and excluded["rgb_mode"] == 3 and excluded["flagged_total"] == 4


def test_leakage_check_raises():
    bad = pd.DataFrame({"group_id": ["g1", "g1"], "split": ["train", "val"]})
    with pytest.raises(ValueError, match="span more than one split"):
        check_no_group_leakage(bad)
