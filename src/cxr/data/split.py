"""Stage 4: build train / val / test splits.

Test is kept exactly as published. Val is carved out of the train pool with
StratifiedGroupKFold: stratified by label, and every group_id (near-duplicates and
images of the same patient, from dedup.py) stays inside a single split.

Writes data/processed/splits.csv and reports/data_quality/split_{summary.json,report.md}.

Usage: uv run python -m cxr.data.split --config configs/data.yaml
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from cxr.utils.config import load_config

SPLITS = ("train", "val", "test")
COLUMNS = ["image_id", "path", "label", "label_name", "split", "source_split", "group_id", "patient_id"]


def build_pool(
    dedup: pd.DataFrame, validation: pd.DataFrame, pairs: pd.DataFrame, cfg: dict
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (train_pool, test, excluded counts) after merging and filtering."""
    scfg = cfg["split"]
    flags = scfg.get("exclude_flags") or []
    df = dedup.merge(validation[["image_id", *flags]], on="image_id", how="left", validate="1:1")
    df = df.rename(columns={"split": "source_split"})

    pool_sources = ["train", "validation"] if scfg["merge_hf_validation"] else ["train"]
    test = df[df["source_split"] == "test"].copy()
    pool = df[df["source_split"].isin(pool_sources)].copy()

    # Per-flag counts are on the full pool (flags overlap); "flagged_total" is the union removed.
    excluded = {}
    if flags:
        flagged = pool[flags].astype(bool)
        excluded = {flag: int(flagged[flag].sum()) for flag in flags}
        excluded["flagged_total"] = int(flagged.any(axis=1).sum())
        pool = pool[~flagged.any(axis=1)]

    if scfg["drop_exact_duplicates"]:
        exact = pairs[pairs["exact"]]
        in_pool = exact["image_id_a"].isin(pool["image_id"]) & exact["image_id_b"].isin(pool["image_id"])
        # Pairs are ordered (a before b), so dropping every "b" keeps the first copy of each set.
        dupes = set(exact.loc[in_pool, "image_id_b"])
        excluded["exact_duplicate"] = len(dupes)
        pool = pool[~pool["image_id"].isin(dupes)]

    return pool.drop(columns=flags), test.drop(columns=flags), excluded


def assign_val(pool: pd.DataFrame, val_fraction: float, seed: int) -> pd.Series:
    """Boolean mask over pool: True for validation rows. Groups never straddle train/val."""
    n_splits = max(2, round(1 / val_fraction))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    _, val_idx = next(sgkf.split(pool, pool["label"], groups=pool["group_id"]))
    mask = pd.Series(False, index=pool.index)
    mask.iloc[val_idx] = True
    return mask


def make_splits(
    dedup: pd.DataFrame, validation: pd.DataFrame, pairs: pd.DataFrame, cfg: dict
) -> tuple[pd.DataFrame, dict]:
    scfg = cfg["split"]
    pool, test, excluded = build_pool(dedup, validation, pairs, cfg)

    is_val = assign_val(pool, scfg["val_fraction"], scfg["seed"])
    pool["split"] = is_val.map({True: "val", False: "train"})
    test["split"] = "test"

    splits = pd.concat([pool, test], ignore_index=True)[COLUMNS]
    check_no_group_leakage(splits)
    return splits, excluded


def check_no_group_leakage(splits: pd.DataFrame) -> None:
    per_group = splits.groupby("group_id")["split"].nunique()
    leaking = per_group[per_group > 1]
    if len(leaking):
        raise ValueError(f"{len(leaking)} group(s) span more than one split, e.g. {list(leaking.index[:5])}")


def summarize(splits: pd.DataFrame, excluded: dict, cfg: dict) -> dict:
    counts = {}
    for split in SPLITS:
        g = splits[splits["split"] == split]
        counts[split] = {
            **{k: int(v) for k, v in g["label_name"].value_counts().sort_index().items()},
            "total": len(g),
            "groups": int(g["group_id"].nunique()),
            "pneumonia_ratio": round(float((g["label_name"] == "PNEUMONIA").mean()), 4),
            "from_source": {k: int(v) for k, v in g["source_split"].value_counts().items()},
        }
    pool_size = counts["train"]["total"] + counts["val"]["total"]
    return {
        "settings": cfg["split"],
        "counts": counts,
        "val_share_of_pool": round(counts["val"]["total"] / pool_size, 4),
        "excluded_from_train_pool": excluded,
        "group_leakage": 0,  # check_no_group_leakage raised otherwise
    }


def render_markdown(s: dict) -> str:
    st = s["settings"]
    lines = [
        "# Split report",
        "",
        (
            f"- `StratifiedGroupKFold`: target val fraction {st['val_fraction']}, seed {st['seed']}; "
            f"actual val share of the train pool: **{s['val_share_of_pool']:.2%}**"
        ),
        f"- HF validation merged into train pool: `{st['merge_hf_validation']}`",
        f"- Excluded from train pool: {s['excluded_from_train_pool'] or 'none'}",
        "- Test is kept as published. No group_id spans more than one split (checked).",
        "",
        "| Split | NORMAL | PNEUMONIA | Total | Groups | Pneumonia ratio | From source split |",
        "| :-- | --: | --: | --: | --: | --: | :-- |",
    ]
    for split, c in s["counts"].items():
        lines.append(
            f"| {split} | {c.get('NORMAL', 0)} | {c.get('PNEUMONIA', 0)} | {c['total']} | {c['groups']} "
            f"| {c['pneumonia_ratio']:.2%} | {c['from_source']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    interim_dir = Path(cfg["interim_dir"])
    dedup = pd.read_csv(interim_dir / "dedup.csv")
    validation = pd.read_csv(interim_dir / "validation.csv")
    # Explicit bool: with zero pairs pandas reads "exact" as object, and pairs[pairs["exact"]] selects columns.
    pairs = pd.read_csv(interim_dir / "near_duplicate_pairs.csv", dtype={"exact": bool})
    splits, excluded = make_splits(dedup, validation, pairs, cfg)

    processed_dir = Path(cfg["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    splits.to_csv(processed_dir / "splits.csv", index=False)

    summary = summarize(splits, excluded, cfg)
    report_dir = Path(cfg["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    (report_dir / "split_report.md").write_text(render_markdown(summary), encoding="utf-8")

    for split, c in summary["counts"].items():
        print(f"  {split:<5} total {c['total']:>5}  pneumonia {c['pneumonia_ratio']:.2%}")
    print(f"Excluded from train pool: {excluded}")
    print(f"Wrote {processed_dir / 'splits.csv'}")


if __name__ == "__main__":
    main()
