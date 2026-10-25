"""Stage 5: reference statistics computed on the TRAIN split only.

- Per-image quality stats (image_stats.py) summarized as quantiles: the reference
  distribution that monitoring compares incoming images against.
- Pixel mean/std after `prepare_gray`, for reporting (the model normalizes with the
  backbone's ImageNet mean/std from configs/data.yaml).

Writes data/reference/stats.json (committed) and data/interim/train_image_stats.csv.

Usage: python -m cxr.data.reference_stats --config configs/data.yaml
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from cxr.data.transforms import prepare_gray
from cxr.monitoring.image_stats import STAT_NAMES, image_stats
from cxr.utils.config import load_config

QUANTILES = (0.005, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.995)


def _one_image(path: str, image_cfg: dict) -> dict:
    """Quality stats plus pixel sums (for an exact pooled mean/std)."""
    with Image.open(path) as img:
        stats = image_stats(img, image_cfg)
        pixels = np.asarray(prepare_gray(img, image_cfg), dtype=np.float64) / 255.0
    stats["_sum"] = float(pixels.sum())
    stats["_sum_sq"] = float((pixels**2).sum())
    stats["_n"] = pixels.size
    return stats


def summarize_stats(stats: pd.DataFrame) -> dict:
    out = {}
    for name in STAT_NAMES:
        col = stats[name]
        out[name] = {
            "mean": float(col.mean()),
            "std": float(col.std()),
            "min": float(col.min()),
            "max": float(col.max()),
            "quantiles": {f"p{q * 100:g}": float(col.quantile(q)) for q in QUANTILES},
        }
    return out


def pixel_mean_std(stats: pd.DataFrame) -> dict:
    n = stats["_n"].sum()
    mean = stats["_sum"].sum() / n
    var = stats["_sum_sq"].sum() / n - mean**2
    return {"mean": float(mean), "std": float(np.sqrt(var))}


def compute_reference(splits: pd.DataFrame, cfg: dict) -> tuple[dict, pd.DataFrame]:
    train = splits[splits["split"] == "train"].reset_index(drop=True)
    fn = partial(_one_image, image_cfg=cfg["image"])
    with ProcessPoolExecutor(max_workers=cfg["reference_stats"].get("workers")) as pool:
        rows = list(pool.map(fn, train["path"], chunksize=32))
    stats = pd.DataFrame(rows)

    reference = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": {"hf_repo": cfg["hf_repo"], "revision": cfg["revision"], "split": "train"},
        "n_images": len(train),
        "image": cfg["image"],
        "pixel": pixel_mean_std(stats),
        "stats": summarize_stats(stats),
        "stats_by_label": {
            label: summarize_stats(stats[train["label_name"] == label])
            for label in sorted(train["label_name"].unique())
        },
    }
    per_image = pd.concat([train[["image_id", "label_name"]], stats[list(STAT_NAMES)]], axis=1)
    return reference, per_image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    splits = pd.read_csv(Path(cfg["processed_dir"]) / "splits.csv")
    reference, per_image = compute_reference(splits, cfg)

    reference_dir = Path(cfg["reference_dir"])
    reference_dir.mkdir(parents=True, exist_ok=True)
    with open(reference_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(reference, f, indent=2)
    per_image.to_csv(Path(cfg["interim_dir"]) / "train_image_stats.csv", index=False)

    print(f"Reference from {reference['n_images']} train images -> {reference_dir / 'stats.json'}")
    print(f"Pixel mean/std: {reference['pixel']['mean']:.4f} / {reference['pixel']['std']:.4f}")
    for name, s in reference["stats"].items():
        q = s["quantiles"]
        print(f"  {name:<16} p1 {q['p1']:.4g}  median {q['p50']:.4g}  p99 {q['p99']:.4g}")


if __name__ == "__main__":
    main()
