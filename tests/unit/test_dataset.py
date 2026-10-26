import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from cxr.data.dataset import XrayDataset, build_dataset
from cxr.data.transforms import build_transforms

CFG = {
    "processed_dir": "data/processed",
    "image": {"size": 32, "resize": "squash", "mean": [0.5] * 3, "std": [0.5] * 3},
    "augment": {"rotation_deg": 10, "scale": [0.9, 1.1], "translate": 0.05},
}

# (split, label, gray value, stored mode)
ROWS = [
    ("train", 0, 0, "L"),
    ("train", 1, 255, "RGB"),
    ("train", 1, 51, "L"),
    ("val", 0, 102, "L"),
    ("test", 1, 204, "L"),
]


@pytest.fixture
def repo(tmp_path):
    """Fake repo root: flat images under data/raw/images and a splits.csv with relative paths."""
    img_dir = tmp_path / "data" / "raw" / "images"
    img_dir.mkdir(parents=True)
    rows = []
    for i, (split, label, value, mode) in enumerate(ROWS):
        rel = f"data/raw/images/{i}.png"
        Image.fromarray(np.full((40, 60), value, dtype=np.uint8)).convert(mode).save(tmp_path / rel)
        rows.append({"image_id": f"i{i}", "path": rel, "label": label, "split": split})
    splits = pd.DataFrame(rows)
    (tmp_path / "data" / "processed").mkdir(parents=True)
    splits.to_csv(tmp_path / "data" / "processed" / "splits.csv", index=False)
    return tmp_path, splits


def test_filters_split_and_keeps_order(repo):
    root, splits = repo
    ds = XrayDataset(splits, "train", build_transforms(CFG, train=False), root=root)
    assert len(ds) == 3
    assert list(ds.frame["image_id"]) == ["i0", "i1", "i2"]
    assert ds.labels == [0, 1, 1]


def test_item_is_normalized_tensor_and_int_label(repo):
    root, splits = repo
    ds = XrayDataset(splits, "train", build_transforms(CFG, train=False), root=root)
    x, y = ds[2]
    assert x.shape == (3, 32, 32) and x.dtype == torch.float32
    assert y == 1 and isinstance(y, int)
    # Flat gray 51 -> 0.2 after scaling -> (0.2 - 0.5) / 0.5 = -0.6 everywhere.
    assert torch.allclose(x, torch.full_like(x, -0.6), atol=1e-5)


def test_rgb_stored_image_matches_gray(repo):
    root, splits = repo
    ds = XrayDataset(splits, "train", build_transforms(CFG, train=False), root=root)
    x, _ = ds[1]  # stored as RGB, value 255 -> (1 - 0.5) / 0.5 = 1
    assert torch.allclose(x, torch.ones_like(x), atol=1e-5)


def test_unknown_or_empty_split_raises(repo):
    root, splits = repo
    t = build_transforms(CFG, train=False)
    with pytest.raises(ValueError, match="Unknown split"):
        XrayDataset(splits, "validation", t, root=root)
    with pytest.raises(ValueError, match="no rows"):
        XrayDataset(splits[splits["split"] != "test"], "test", t, root=root)


def test_build_dataset_reads_csv_and_augments_train_only(repo):
    root, _ = repo
    train, val, test = (build_dataset(CFG, s, root=root) for s in ("train", "val", "test"))
    assert (len(train), len(val), len(test)) == (3, 1, 1)
    assert val.labels == [0] and test.labels == [1]
    # RandomAffine is the only random step, so its presence marks the train pipeline.
    assert "RandomAffine" in repr(train.transform)
    assert "RandomAffine" not in repr(val.transform) and "RandomAffine" not in repr(test.transform)


def test_dataloader_batches_with_workers(repo):
    root, _ = repo
    # spawn (not Linux's default fork) pickles the dataset, as on Windows / macOS.
    loader = DataLoader(
        build_dataset(CFG, "train", root=root), batch_size=2, num_workers=2, multiprocessing_context="spawn"
    )
    batches = list(loader)
    assert [b[0].shape for b in batches] == [(2, 3, 32, 32), (1, 3, 32, 32)]
    assert torch.cat([b[1] for b in batches]).tolist() == [0, 1, 1]
