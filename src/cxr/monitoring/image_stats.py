"""Per-image quality statistics, shared by reference_stats.py (train) and the API (monitoring).

Size and JPEG blockiness are measured on the original image (8x8 JPEG blocks vanish when
downsized); intensity, sharpness and noise are measured after `prepare_gray`, so they do
not depend on the original resolution.
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
    "noise",
    "blockiness",
)

JPEG_BLOCK = 8


def laplacian_variance(gray: np.ndarray) -> float:
    """Variance of the 4-neighbour Laplacian; low values mean a blurry image."""
    g = gray.astype(np.float64)
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())


def noise_sigma(gray: np.ndarray) -> float:
    """Noise standard deviation estimate (Immerkaer 1996): mean |3x3 Laplacian-difference| scaled.

    The kernel cancels smooth intensity changes, so on smooth X-rays mostly noise remains.
    """
    g = gray.astype(np.float64)
    conv = (
        g[:-2, :-2] - 2 * g[:-2, 1:-1] + g[:-2, 2:]
        - 2 * g[1:-1, :-2] + 4 * g[1:-1, 1:-1] - 2 * g[1:-1, 2:]
        + g[2:, :-2] - 2 * g[2:, 1:-1] + g[2:, 2:]
    )  # fmt: skip
    return float(np.sqrt(np.pi / 2) * np.abs(conv).mean() / 6)


def _block_edge_ratio(diff: np.ndarray) -> float:
    """Mean |diff| across 8-px block edges / mean |diff| elsewhere; diff[:, j] = col j+1 - col j."""
    edge = np.arange(diff.shape[1]) % JPEG_BLOCK == JPEG_BLOCK - 1
    if not edge.any() or edge.all():
        return 1.0
    # Sums instead of diff[:, ~edge].mean(): no copy of 7/8 of a possibly 16 MP array. Exact for ints.
    total = float(diff.sum(dtype=np.float64))
    edge_sum = float(diff[:, edge].sum(dtype=np.float64))
    edge_mean = edge_sum / (diff.shape[0] * edge.sum())
    inside_mean = (total - edge_sum) / (diff.shape[0] * (~edge).sum())
    # eps: a flat image gives 1; flat blocks with steps only at their edges give a very large value.
    eps = 1e-6
    return float((edge_mean + eps) / (inside_mean + eps))


def blockiness(gray: np.ndarray) -> float:
    """JPEG blocking on the original image: ~1 without 8x8 block edges, larger with stronger compression.

    gray: 8-bit values. int16 keeps memory at 2 bytes per pixel per array (API images can be large).
    """
    g = gray.astype(np.int16)
    horizontal = _block_edge_ratio(np.abs(np.diff(g, axis=1)))
    vertical = _block_edge_ratio(np.abs(np.diff(g, axis=0)).T)
    return (horizontal + vertical) / 2


def image_stats(img: Image.Image, image_cfg: dict) -> dict[str, float]:
    w, h = img.size
    gray = np.asarray(prepare_gray(img, image_cfg), dtype=np.float64) / 255.0
    original = np.asarray(img.convert("L"))
    return {
        "width": float(w),
        "height": float(h),
        "aspect_ratio": w / h,
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "blur": laplacian_variance(gray),
        "dark_fraction": float((gray < 0.05).mean()),
        "bright_fraction": float((gray > 0.95).mean()),
        "noise": noise_sigma(gray),
        "blockiness": blockiness(original),
    }
