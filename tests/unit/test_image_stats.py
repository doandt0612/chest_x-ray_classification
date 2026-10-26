import numpy as np
from PIL import Image, ImageFilter

from cxr.monitoring.image_stats import STAT_NAMES, image_stats, laplacian_variance

IMAGE_CFG = {"size": 224, "resize": "squash"}


def _stripes(w=600, h=400):
    arr = np.zeros((h, w), dtype=np.uint8)
    arr[:, ::8] = 255
    return Image.fromarray(arr)


def test_all_stats_present_and_size_is_original():
    s = image_stats(_stripes(600, 400), IMAGE_CFG)
    assert set(s) == set(STAT_NAMES)
    assert (s["width"], s["height"]) == (600, 400)
    assert s["aspect_ratio"] == 1.5


def test_flat_image():
    s = image_stats(Image.fromarray(np.full((300, 300), 128, dtype=np.uint8)), IMAGE_CFG)
    assert abs(s["brightness"] - 128 / 255) < 1e-9
    assert s["contrast"] == 0 and s["blur"] == 0
    assert s["dark_fraction"] == 0 and s["bright_fraction"] == 0


def test_dark_image_is_darker():
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (300, 300)).astype(np.uint8)
    dark = (base * 0.3).astype(np.uint8)
    s_base, s_dark = (
        image_stats(Image.fromarray(base), IMAGE_CFG),
        image_stats(Image.fromarray(dark), IMAGE_CFG),
    )
    assert s_dark["brightness"] < s_base["brightness"]
    assert s_dark["dark_fraction"] > s_base["dark_fraction"]


def test_blur_lowers_laplacian_variance():
    sharp = _stripes()
    blurred = sharp.filter(ImageFilter.GaussianBlur(4))
    assert image_stats(blurred, IMAGE_CFG)["blur"] < image_stats(sharp, IMAGE_CFG)["blur"] / 5


def test_stats_identical_for_rgb_storage():
    img = _stripes()
    assert image_stats(img, IMAGE_CFG) == image_stats(img.convert("RGB"), IMAGE_CFG)


def test_laplacian_of_linear_ramp_is_zero():
    ramp = np.tile(np.arange(50, dtype=np.float64), (50, 1))
    assert laplacian_variance(ramp) == 0
