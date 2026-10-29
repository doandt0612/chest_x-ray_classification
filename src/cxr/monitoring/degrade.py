"""Synthetic degradations with known strength, to test the input quality check (D10).

Each degradation works on the original image in 8-bit gray, as an X-ray would arrive at the API:
- blur: Gaussian blur, level = radius (std) in pixels
- noise: additive Gaussian noise, level = std in gray levels (0-255)
- dark: pixel values multiplied by level (< 1 darkens)
- jpeg: re-encoded as JPEG, level = quality (lower is stronger)

Levels are set in configs/monitor.yaml, `degradations`, ordered from mild to strongest.
"""

import io

import numpy as np
from PIL import Image, ImageFilter

KINDS = ("blur", "noise", "dark", "jpeg")


def degrade(img: Image.Image, kind: str, level: float, rng: np.random.Generator) -> Image.Image:
    """Return a degraded 8-bit gray copy of img; rng is used by noise only."""
    gray = img.convert("L")
    if kind == "blur":
        return gray.filter(ImageFilter.GaussianBlur(radius=level))
    if kind == "noise":
        arr = np.asarray(gray, dtype=np.float64) + rng.normal(0.0, level, size=(gray.height, gray.width))
        return Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8))
    if kind == "dark":
        arr = np.asarray(gray, dtype=np.float64) * level
        return Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8))
    if kind == "jpeg":
        buf = io.BytesIO()
        gray.save(buf, format="JPEG", quality=int(level))
        out = Image.open(buf)
        out.load()
        return out
    raise ValueError(f"Unknown degradation {kind!r}; expected one of {KINDS}")
