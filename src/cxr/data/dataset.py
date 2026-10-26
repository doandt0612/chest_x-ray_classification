"""torch Dataset over one split of data/processed/splits.csv.

Each item is (image tensor, label): the image goes through the shared transforms
(transforms.py), the label is 0 = NORMAL, 1 = PNEUMONIA. Rows keep the CSV order,
so predictions from a non-shuffled DataLoader line up with `dataset.frame`.
"""

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from cxr.data.split import SPLITS
from cxr.data.transforms import build_transforms


class XrayDataset(Dataset):
    def __init__(
        self,
        splits: pd.DataFrame,
        split: str,
        transform: Callable[[Image.Image], torch.Tensor],
        root: str | Path = ".",
    ):
        """splits: the splits.csv table. root: directory the CSV `path` column is relative to."""
        if split not in SPLITS:
            raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}")
        self.frame = splits[splits["split"] == split].reset_index(drop=True)
        if self.frame.empty:
            raise ValueError(f"Split {split!r} has no rows")
        self.root = Path(root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.frame.iloc[idx]
        with Image.open(self.root / row["path"]) as img:
            img.load()  # decode now: PIL opens lazily and the file closes when the block ends
            x = self.transform(img)
        return x, int(row["label"])

    @property
    def labels(self) -> list[int]:
        """All labels in row order, e.g. for class weights; does not open any image."""
        return self.frame["label"].astype(int).tolist()


def build_dataset(cfg: dict, split: str, root: str | Path = ".") -> XrayDataset:
    """Dataset for `split` from <processed_dir>/splits.csv; augmentation only for train."""
    splits = pd.read_csv(Path(root) / cfg["processed_dir"] / "splits.csv")
    transform = build_transforms(cfg, train=(split == "train"))
    return XrayDataset(splits, split, transform, root=root)
