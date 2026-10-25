"""Stage 2: check every image in manifest.csv.

Checks: image opens and decodes, valid label, size, aspect ratio, color mode
(and whether an RGB image is actually grayscale), label distribution per split.
Writes data/interim/validation.csv and reports/data_quality/validation_{summary.json,report.md}.

Usage: python -m cxr.data.validate --config configs/data.yaml
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from cxr.utils.config import load_config

# rgb_mode: stored as RGB. In this dataset all 283 such images are PNEUMONIA crops of the lung
# field at low resolution (a separate cohort), so the flag lets split.py exclude them.
FLAGS = ("corrupt", "label_invalid", "too_small", "extreme_aspect", "color", "rgb_mode")


def check_image(path: str, min_side: int, max_aspect_ratio: float) -> dict:
    """Open and decode one image; return its properties and quality flags."""
    result = {
        "format": None,
        "mode": None,
        "width": None,
        "height": None,
        "is_gray_content": None,
        "error": None,
        "corrupt": False,
        "too_small": False,
        "extreme_aspect": False,
        "color": False,
        "rgb_mode": False,
    }
    try:
        with Image.open(path) as img:
            img.verify()  # structural check; the image must be reopened afterwards
        with Image.open(path) as img:
            img.load()  # full decode catches truncated files that verify() misses
            result["format"] = img.format
            result["mode"] = img.mode
            result["width"], result["height"] = img.size
            result["is_gray_content"] = _is_gray_content(img)
    except Exception as e:  # noqa: BLE001 - any decode failure means the image is unusable
        result["corrupt"] = True
        result["error"] = f"{type(e).__name__}: {e}"
        return result

    w, h = result["width"], result["height"]
    result["too_small"] = min(w, h) < min_side
    result["extreme_aspect"] = max(w, h) / min(w, h) > max_aspect_ratio
    result["color"] = not result["is_gray_content"]
    result["rgb_mode"] = result["mode"] == "RGB"
    return result


def _is_gray_content(img: Image.Image) -> bool:
    """True if the image carries no color: mode L/I/F, or RGB with (near-)identical channels."""
    if img.mode in ("1", "L", "LA", "I", "I;16", "F"):
        return True
    small = np.asarray(img.convert("RGB").resize((128, 128)), dtype=np.int16)
    spread = np.abs(small[..., 0] - small[..., 1]).max(), np.abs(small[..., 1] - small[..., 2]).max()
    return max(spread) <= 2  # tolerate JPEG chroma noise


def validate(manifest: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    vcfg = cfg["validate"]
    label_names = {int(k): v for k, v in cfg["label_names"].items()}
    fn = partial(check_image, min_side=vcfg["min_side"], max_aspect_ratio=vcfg["max_aspect_ratio"])
    with ProcessPoolExecutor(max_workers=vcfg.get("workers")) as pool:
        checks = list(pool.map(fn, manifest["path"], chunksize=32))

    df = pd.concat([manifest.reset_index(drop=True), pd.DataFrame(checks)], axis=1)
    df["label_invalid"] = ~df["label"].isin(label_names.keys()) | (
        df["label"].map(label_names) != df["label_name"]
    )
    # Size, aspect and color are warnings; only corrupt images and bad labels are unusable.
    df["usable"] = ~(df["corrupt"] | df["label_invalid"])
    return df


def summarize(df: pd.DataFrame, cfg: dict) -> dict:
    ok = df[~df["corrupt"]]
    summary = {
        "n_images": len(df),
        "n_usable": int(df["usable"].sum()),
        "thresholds": cfg["validate"],
        "flags": {flag: int(df[flag].sum()) for flag in FLAGS},
        "flags_by_split": {
            split: {flag: int(g[flag].sum()) for flag in FLAGS} for split, g in df.groupby("split")
        },
        "label_distribution": {
            split: {
                **{k: int(v) for k, v in g["label_name"].value_counts().items()},
                "pneumonia_ratio": round(float((g["label_name"] == "PNEUMONIA").mean()), 4),
            }
            for split, g in df.groupby("split")
        },
        "modes": {k: int(v) for k, v in ok["mode"].value_counts().items()},
        "formats": {k: int(v) for k, v in ok["format"].value_counts().items()},
        "size": {
            dim: {
                "min": int(ok[dim].min()),
                "p05": int(ok[dim].quantile(0.05)),
                "median": int(ok[dim].median()),
                "p95": int(ok[dim].quantile(0.95)),
                "max": int(ok[dim].max()),
            }
            for dim in ("width", "height")
        },
        "corrupt_images": df.loc[df["corrupt"], ["image_id", "path", "error"]].to_dict("records"),
    }
    return summary


def render_markdown(s: dict) -> str:
    lines = [
        "# Data validation report",
        "",
        f"- Images checked: **{s['n_images']}**, usable: **{s['n_usable']}**",
        (
            f"- Thresholds: `min_side={s['thresholds']['min_side']}`, "
            f"`max_aspect_ratio={s['thresholds']['max_aspect_ratio']}`"
        ),
        "",
        "## Label distribution",
        "",
        "| Split | NORMAL | PNEUMONIA | Pneumonia ratio |",
        "| :-- | --: | --: | --: |",
    ]
    for split, d in s["label_distribution"].items():
        lines.append(
            f"| {split} | {d.get('NORMAL', 0)} | {d.get('PNEUMONIA', 0)} | {d['pneumonia_ratio']:.2%} |"
        )
    lines += ["", "## Flags", "", "| Split | " + " | ".join(FLAGS) + " |", "| :-- |" + " --: |" * len(FLAGS)]
    for split, f in s["flags_by_split"].items():
        lines.append(f"| {split} | " + " | ".join(str(f[k]) for k in FLAGS) + " |")
    lines.append("| **total** | " + " | ".join(f"**{s['flags'][k]}**" for k in FLAGS) + " |")
    lines += [
        "",
        "`corrupt` and `label_invalid` make an image unusable; the other flags are warnings.",
        "",
        "## Image properties",
        "",
        f"- Color modes: {s['modes']}",
        f"- Formats: {s['formats']}",
        "",
        "| | min | p05 | median | p95 | max |",
        "| :-- | --: | --: | --: | --: | --: |",
    ]
    for dim, d in s["size"].items():
        lines.append(f"| {dim} | {d['min']} | {d['p05']} | {d['median']} | {d['p95']} | {d['max']} |")
    if s["corrupt_images"]:
        lines += ["", "## Corrupt images", ""]
        lines += [f"- `{c['image_id']}`: {c['error']}" for c in s["corrupt_images"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    interim_dir = Path(cfg["interim_dir"])
    manifest = pd.read_csv(interim_dir / "manifest.csv")
    df = validate(manifest, cfg)
    df.to_csv(interim_dir / "validation.csv", index=False)

    summary = summarize(df, cfg)
    report_dir = Path(cfg["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / "validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    (report_dir / "validation_report.md").write_text(render_markdown(summary), encoding="utf-8")

    print(f"Checked {summary['n_images']} images, usable {summary['n_usable']}")
    print(f"Flags: {summary['flags']}")
    print(f"Report: {report_dir / 'validation_report.md'}")


if __name__ == "__main__":
    main()
