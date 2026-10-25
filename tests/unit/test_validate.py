import numpy as np
import pandas as pd
from PIL import Image

from cxr.data.validate import check_image, validate

CFG = {
    "label_names": {0: "NORMAL", 1: "PNEUMONIA"},
    "validate": {"min_side": 256, "max_aspect_ratio": 2.5, "workers": 1},
}


def _save(tmp_path, name, array, mode=None):
    path = tmp_path / name
    Image.fromarray(array, mode=mode).save(path)
    return str(path)


def test_gray_image_passes(tmp_path):
    path = _save(tmp_path, "gray.png", np.full((300, 400), 128, dtype=np.uint8))
    r = check_image(path, min_side=256, max_aspect_ratio=2.5)
    assert not r["corrupt"] and not r["too_small"] and not r["extreme_aspect"]
    assert r["mode"] == "L" and r["is_gray_content"] and not r["color"] and not r["rgb_mode"]
    assert (r["width"], r["height"]) == (400, 300)


def test_rgb_with_equal_channels_is_gray_content(tmp_path):
    gray = np.random.default_rng(0).integers(0, 255, (300, 300), dtype=np.uint8)
    path = _save(tmp_path, "rgb_gray.png", np.stack([gray] * 3, axis=-1))
    r = check_image(path, min_side=256, max_aspect_ratio=2.5)
    assert r["mode"] == "RGB" and r["is_gray_content"] and not r["color"] and r["rgb_mode"]


def test_color_image_is_flagged(tmp_path):
    arr = np.zeros((300, 300, 3), dtype=np.uint8)
    arr[..., 0] = 200
    r = check_image(_save(tmp_path, "red.png", arr), min_side=256, max_aspect_ratio=2.5)
    assert r["color"]


def test_small_and_extreme_aspect_are_flagged(tmp_path):
    path = _save(tmp_path, "thin.png", np.zeros((100, 400), dtype=np.uint8))
    r = check_image(path, min_side=256, max_aspect_ratio=2.5)
    assert r["too_small"] and r["extreme_aspect"]


def test_corrupt_files_are_flagged(tmp_path):
    garbage = tmp_path / "garbage.jpeg"
    garbage.write_bytes(b"not an image")

    good = tmp_path / "good.jpeg"
    Image.fromarray(np.full((300, 300), 100, dtype=np.uint8)).save(good)
    truncated = tmp_path / "truncated.jpeg"
    truncated.write_bytes(good.read_bytes()[:200])

    for path in (garbage, truncated):
        r = check_image(str(path), min_side=256, max_aspect_ratio=2.5)
        assert r["corrupt"] and r["error"]


def test_validate_marks_bad_labels_unusable(tmp_path):
    path = _save(tmp_path, "img.png", np.full((300, 300), 50, dtype=np.uint8))
    manifest = pd.DataFrame(
        {
            "image_id": ["a", "b", "c"],
            "split": ["train"] * 3,
            "label": [0, 1, 7],
            "label_name": ["NORMAL", "NORMAL", "UNKNOWN"],
            "path": [path] * 3,
        }
    )
    df = validate(manifest, CFG)
    assert df["label_invalid"].tolist() == [False, True, True]
    assert df["usable"].tolist() == [True, False, False]
