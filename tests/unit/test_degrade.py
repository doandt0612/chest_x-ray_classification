import numpy as np
import pytest
from PIL import Image

from cxr.monitoring.degrade import KINDS, degrade
from cxr.monitoring.image_stats import blockiness, laplacian_variance, noise_sigma


def _xray_like(w=400, h=300, seed=0):
    """Smooth gradient plus fine texture, roughly like an X-ray."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    arr = 60 + 120 * np.exp(-(((x - w / 2) / (w / 3)) ** 2 + ((y - h / 2) / (h / 3)) ** 2))
    arr += rng.normal(0, 3, size=(h, w))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _rng():
    return np.random.default_rng(0)


@pytest.mark.parametrize("kind,level", [("blur", 2), ("noise", 10), ("dark", 0.5), ("jpeg", 20)])
def test_keeps_size_and_returns_gray(kind, level):
    img = _xray_like().convert("RGB")
    out = degrade(img, kind, level, _rng())
    assert out.size == img.size
    assert out.mode == "L"


def test_does_not_modify_input():
    img = _xray_like()
    before = np.asarray(img).copy()
    for kind, level in [("blur", 4), ("noise", 20), ("dark", 0.3), ("jpeg", 10)]:
        degrade(img, kind, level, _rng())
    assert (np.asarray(img) == before).all()


def test_dark_multiplies_pixels():
    img = Image.fromarray(np.full((10, 10), 200, dtype=np.uint8))
    assert (np.asarray(degrade(img, "dark", 0.3, _rng())) == 60).all()


def test_noise_is_seeded_and_clipped():
    img = Image.fromarray(np.full((50, 50), 250, dtype=np.uint8))
    a = np.asarray(degrade(img, "noise", 40, np.random.default_rng(1)))
    b = np.asarray(degrade(img, "noise", 40, np.random.default_rng(1)))
    c = np.asarray(degrade(img, "noise", 40, np.random.default_rng(2)))
    assert (a == b).all() and (a != c).any()
    assert a.max() == 255  # clipped, not wrapped around


def test_stronger_level_moves_its_stat_further():
    img = _xray_like()
    gray = lambda kind, level: np.asarray(degrade(img, kind, level, _rng()), dtype=np.float64)
    assert laplacian_variance(gray("blur", 8)) < laplacian_variance(gray("blur", 1))
    assert noise_sigma(gray("noise", 40)) > noise_sigma(gray("noise", 5))
    assert gray("dark", 0.3).mean() < gray("dark", 0.9).mean()
    assert blockiness(gray("jpeg", 10).astype(np.uint8)) > blockiness(gray("jpeg", 50).astype(np.uint8))


def test_jpeg_output_is_a_decoded_jpeg():
    out = degrade(_xray_like(), "jpeg", 30, _rng())
    assert out.format == "JPEG"


def test_unknown_kind_raises():
    with pytest.raises(ValueError, match="Unknown degradation"):
        degrade(_xray_like(), "rotate", 1, _rng())
    assert KINDS == ("blur", "noise", "dark", "jpeg")
