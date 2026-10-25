"""Stage 1: download the dataset at the pinned revision, extract images, write manifest.csv.

Usage: python -m cxr.data.download --config configs/data.yaml
"""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

from cxr.utils.config import load_config

SPLITS = ("train", "validation", "test")


def download(cfg: dict) -> Path:
    path = snapshot_download(
        cfg["hf_repo"],
        repo_type="dataset",
        revision=cfg["revision"],
        local_dir=cfg["raw_dir"],
    )
    return Path(path)


def extract_images(cfg: dict) -> pd.DataFrame:
    """Write each image's bytes to images_dir/<split>/<image_id><ext>; return the manifest."""
    raw_dir = Path(cfg["raw_dir"])
    images_dir = Path(cfg["images_dir"])
    label_names = {int(k): v for k, v in cfg["label_names"].items()}

    rows = []
    for split in SPLITS:
        files = sorted((raw_dir / "data").glob(f"{split}-*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files for split '{split}' in {raw_dir / 'data'}")
        out_dir = images_dir / split
        out_dir.mkdir(parents=True, exist_ok=True)

        idx = 0
        for f in files:
            for batch in pq.ParquetFile(f).iter_batches(batch_size=256):
                for rec in batch.to_pylist():
                    data = rec["image"]["bytes"]
                    source_filename = rec["image"]["path"]
                    ext = Path(source_filename).suffix.lower() or ".jpeg"
                    image_id = f"{split}_{idx:05d}"
                    out_path = out_dir / f"{image_id}{ext}"
                    if not (out_path.exists() and out_path.stat().st_size == len(data)):
                        out_path.write_bytes(data)
                    rows.append(
                        {
                            "image_id": image_id,
                            "split": split,
                            "label": rec["label"],
                            "label_name": label_names.get(rec["label"], "UNKNOWN"),
                            "path": out_path.as_posix(),
                            "source_filename": source_filename,
                            "sha256": hashlib.sha256(data).hexdigest(),
                            "n_bytes": len(data),
                        }
                    )
                    idx += 1
    return pd.DataFrame(rows)


def count_images(manifest: pd.DataFrame) -> dict:
    counts = manifest.groupby(["split", "label_name"]).size().unstack(fill_value=0)
    return {
        split: {**{k: int(v) for k, v in counts.loc[split].items()}, "total": int(counts.loc[split].sum())}
        for split in counts.index
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    download(cfg)
    manifest = extract_images(cfg)

    interim_dir = Path(cfg["interim_dir"])
    interim_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(interim_dir / "manifest.csv", index=False)

    counts = count_images(manifest)
    report_dir = Path(cfg["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / "counts.json", "w", encoding="utf-8") as f:
        json.dump({"hf_repo": cfg["hf_repo"], "revision": cfg["revision"], "counts": counts}, f, indent=2)

    print(f"Wrote {len(manifest)} rows to {interim_dir / 'manifest.csv'}")
    for split, c in counts.items():
        print(f"  {split:<10} {c}")


if __name__ == "__main__":
    main()
