import numpy as np
import pandas as pd
from PIL import Image

from cxr.data.reference_stats import compute_reference

CFG = {
    "hf_repo": "x/y",
    "revision": "abc",
    "image": {"size": 32, "resize": "squash", "mean": [0.5] * 3, "std": [0.5] * 3},
    "reference_stats": {"workers": 1},
}


def test_reference_uses_train_only(tmp_path):
    rows = []
    for i, (split, value) in enumerate([("train", 50), ("train", 100), ("val", 250), ("test", 250)]):
        p = tmp_path / f"{i}.png"
        Image.fromarray(np.full((40, 60), value, dtype=np.uint8)).save(p)
        rows.append({"image_id": f"i{i}", "path": str(p), "split": split, "label_name": "NORMAL"})
    reference, per_image = compute_reference(pd.DataFrame(rows), CFG)

    assert reference["n_images"] == 2 and list(per_image["image_id"]) == ["i0", "i1"]
    expected_mean = (50 + 100) / 2 / 255
    assert abs(reference["pixel"]["mean"] - expected_mean) < 1e-9
    assert abs(reference["pixel"]["std"] - 25 / 255) < 1e-9
    b = reference["stats"]["brightness"]
    assert abs(b["min"] - 50 / 255) < 1e-9 and abs(b["max"] - 100 / 255) < 1e-9
    assert reference["stats"]["width"]["quantiles"]["p50"] == 60
    assert set(reference["stats_by_label"]) == {"NORMAL"}
