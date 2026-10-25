"""Image preprocessing shared by training, the API and monitoring.

Every consumer goes through `prepare_gray` so the resize policy is defined once:
grayscale ("L", which also normalizes the RGB-stored images) then a fixed-size resize.
`build_transforms` adds augmentation (train only), tensor conversion and normalization.
"""

from collections.abc import Callable

import torch
from PIL import Image
from torchvision.transforms import v2

RESIZE_MODES = ("squash", "center_crop")


def prepare_gray(img: Image.Image, image_cfg: dict) -> Image.Image:
    """Grayscale + resize to image_cfg['size'] square, following image_cfg['resize']."""
    size, mode = image_cfg["size"], image_cfg["resize"]
    if mode not in RESIZE_MODES:
        raise ValueError(f"Unknown resize mode {mode!r}; expected one of {RESIZE_MODES}")

    gray = img.convert("L")
    if mode == "squash":
        return gray.resize((size, size), Image.BILINEAR)

    w, h = gray.size
    scale = size / min(w, h)
    new_w, new_h = max(size, round(w * scale)), max(size, round(h * scale))
    resized = gray.resize((new_w, new_h), Image.BILINEAR)
    left, top = (new_w - size) // 2, (new_h - size) // 2
    return resized.crop((left, top, left + size, top + size))


class PrepareGray3:
    """prepare_gray, repeated to 3 channels for ImageNet-pretrained backbones.

    A class rather than a lambda so the transform pickles for DataLoader workers.
    """

    def __init__(self, image_cfg: dict):
        self.image_cfg = image_cfg

    def __call__(self, img: Image.Image) -> Image.Image:
        return prepare_gray(img, self.image_cfg).convert("RGB")


def build_transforms(cfg: dict, train: bool) -> Callable[[Image.Image], torch.Tensor]:
    """PIL image -> normalized float tensor of shape (3, size, size)."""
    image_cfg = cfg["image"]
    steps: list = [PrepareGray3(image_cfg)]
    if train:
        aug = cfg["augment"]
        steps.append(
            v2.RandomAffine(
                degrees=aug["rotation_deg"],
                translate=(aug["translate"], aug["translate"]),
                scale=tuple(aug["scale"]),
                interpolation=v2.InterpolationMode.BILINEAR,
                fill=0,  # black, like the X-ray background
            )
        )
    steps += [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=image_cfg["mean"], std=image_cfg["std"]),
    ]
    return v2.Compose(steps)
