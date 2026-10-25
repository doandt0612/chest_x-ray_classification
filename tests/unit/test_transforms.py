import pickle

import numpy as np
import pytest
import torch
from PIL import Image

from cxr.data.transforms import build_transforms, prepare_gray

CFG = {
    "image": {
        "size": 224,
        "resize": "squash",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
    "augment": {"rotation_deg": 10, "scale": [0.9, 1.1], "translate": 0.05},
}


def _img(w=600, h=400, seed=0):
    return Image.fromarray(np.random.default_rng(seed).integers(0, 255, (h, w), dtype=np.uint8))


@pytest.mark.parametrize("resize", ["squash", "center_crop"])
def test_prepare_gray_output_size_and_mode(resize):
    out = prepare_gray(_img(), {**CFG["image"], "resize": resize})
    assert out.mode == "L" and out.size == (224, 224)


def test_center_crop_keeps_center():
    arr = np.zeros((200, 600), dtype=np.uint8)
    arr[:, 200:400] = 255  # bright middle third
    out = np.asarray(prepare_gray(Image.fromarray(arr), {**CFG["image"], "resize": "center_crop"}))
    assert out[:, 20:-20].min() > 200  # center crop of a 3:1 image is the middle third


def test_unknown_resize_mode_raises():
    with pytest.raises(ValueError, match="Unknown resize mode"):
        prepare_gray(_img(), {**CFG["image"], "resize": "pad"})


def test_eval_transform_shape_and_determinism():
    t = build_transforms(CFG, train=False)
    a, b = t(_img()), t(_img())
    assert a.shape == (3, 224, 224) and a.dtype == torch.float32
    assert torch.equal(a, b)
    mean = torch.tensor(CFG["image"]["mean"]).view(3, 1, 1)
    std = torch.tensor(CFG["image"]["std"]).view(3, 1, 1)
    raw = a * std + mean  # undo normalization: every channel holds the same gray image
    assert torch.allclose(raw[0], raw[1], atol=1e-6) and torch.allclose(raw[1], raw[2], atol=1e-6)


def test_rgb_and_gray_storage_give_same_tensor():
    gray = _img()
    t = build_transforms(CFG, train=False)
    assert torch.allclose(t(gray), t(gray.convert("RGB")))


def test_normalization_values():
    flat = Image.fromarray(np.full((300, 300), 255, dtype=np.uint8))
    x = build_transforms(CFG, train=False)(flat)
    expected = (1 - torch.tensor(CFG["image"]["mean"])) / torch.tensor(CFG["image"]["std"])
    assert torch.allclose(x.mean(dim=(1, 2)), expected, atol=1e-5)


def test_train_transform_augments_but_keeps_shape():
    torch.manual_seed(0)
    t = build_transforms(CFG, train=True)
    a, b = t(_img()), t(_img())
    assert a.shape == b.shape == (3, 224, 224)
    assert not torch.equal(a, b)


def test_transforms_pickle_for_dataloader_workers():
    t = build_transforms(CFG, train=True)
    restored = pickle.loads(pickle.dumps(t))
    assert restored(_img()).shape == (3, 224, 224)
