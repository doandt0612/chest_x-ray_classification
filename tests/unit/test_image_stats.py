import io

import numpy as np
from PIL import Image, ImageFilter

from cxr.monitoring.image_stats import (
    STAT_NAMES,
    blockiness,
    image_stats,
    laplacian_variance,
    noise_sigma,
)

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


def test_noise_sigma_recovers_gaussian_noise_level():
    # Immerkaer's estimator is unbiased for Gaussian noise on a flat image.
    rng = np.random.default_rng(0)
    flat = np.full((400, 400), 0.5)
    assert noise_sigma(flat) == 0
    assert abs(noise_sigma(flat + rng.normal(0, 0.02, flat.shape)) - 0.02) < 0.001


def _smooth_gradient(w=640, h=480):
    x = np.linspace(0, 1, w)[None, :] * np.linspace(0.3, 1, h)[:, None]
    return Image.fromarray((x * 200 + 20).astype(np.uint8))


def _jpeg(img, quality):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf)


def test_blockiness_near_one_without_jpeg_and_high_with_strong_jpeg():
    rng = np.random.default_rng(0)
    noisy = np.asarray(_smooth_gradient(), dtype=np.float64) + rng.normal(0, 2, (480, 640))
    plain = np.clip(noisy, 0, 255).astype(np.uint8)
    assert abs(blockiness(plain) - 1.0) < 0.05
    assert blockiness(np.full((50, 50), 7)) == 1.0  # flat: no edges at all
    assert blockiness(np.asarray(_jpeg(_smooth_gradient(), 5))) > 2


def test_noise_and_blockiness_in_image_stats():
    img = _smooth_gradient()
    s, s_jpeg = image_stats(img, IMAGE_CFG), image_stats(_jpeg(img, 5), IMAGE_CFG)
    assert s_jpeg["blockiness"] > s["blockiness"]
