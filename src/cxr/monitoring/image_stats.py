"""Per-image quality statistics, shared by reference_stats.py (train) and the API (monitoring).

Size is measured on the original image; intensity and sharpness are measured after
`prepare_gray`, so they do not depend on the original resolution.
"""

import numpy as np
from PIL import Image

from cxr.data.transforms import prepare_gray

STAT_NAMES = (
    "width",
    "height",
    "aspect_ratio",
    "brightness",
    "contrast",
    "blur",
    "dark_fraction",
    "bright_fraction",
)


def laplacian_variance(gray: np.ndarray) -> float:
    """Variance of the 4-neighbour Laplacian; low values mean a blurry image."""
    g = gray.astype(np.float64)
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())


def image_stats(img: Image.Image, image_cfg: dict) -> dict[str, float]:
    w, h = img.size
    gray = np.asarray(prepare_gray(img, image_cfg), dtype=np.float64) / 255.0
    return {
        "width": float(w),
        "height": float(h),
        "aspect_ratio": w / h,
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "blur": laplacian_variance(gray),
        "dark_fraction": float((gray < 0.05).mean()),
        "bright_fraction": float((gray > 0.95).mean()),
    }
