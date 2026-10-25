"""Stage 3: find duplicate / near-duplicate images and assign a group_id.

Two images end up in the same group if (a) their perceptual hashes are within
`max_hamming` bits, or (b) they share the patient ID encoded in the source filename
(when `group_by_patient` is on). Groups are transitive. split.py keeps each group in
one split so near-identical images cannot leak between train and validation.

Writes data/interim/{dedup.csv, near_duplicate_pairs.csv} and
reports/data_quality/dedup_{summary.json,report.md}.

Usage: python -m cxr.data.dedup --config configs/data.yaml
"""

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

from cxr.utils.config import load_config

# person123_bacteria_45.jpeg -> person123_bacteria; IM-0115-0001.jpeg -> IM-115;
# NORMAL2-IM-0383-0001.jpeg -> NORMAL2-IM-383 (separate numbering from IM-).
# personN is numbered separately for bacteria and virus: no personN shares the same kind
# between train and test, and half of train personN mix both kinds, so N alone is not a patient.
_PNEUMONIA_ID = re.compile(r"^person(\d+)_(bacteria|virus)_")
_NORMAL_ID = re.compile(r"^(?:(NORMAL\d+)-)?IM-(\d+)-")


def patient_id(source_filename: str) -> str | None:
    m = _PNEUMONIA_ID.match(source_filename)
    if m:
        return f"person{int(m[1])}_{m[2]}"
    m = _NORMAL_ID.match(source_filename)
    if m:
        prefix = f"{m[1]}-" if m[1] else ""
        return f"{prefix}IM-{int(m[2])}"
    return None


def compute_phash(path: str, hash_size: int) -> str:
    with Image.open(path) as img:
        return str(imagehash.phash(img.convert("L"), hash_size=hash_size))


def _pack(hex_hashes: list[str]) -> np.ndarray:
    """Hex pHash strings -> (n, words) uint64 array for fast XOR + popcount."""
    n_bytes = len(hex_hashes[0]) // 2
    pad = (-n_bytes) % 8
    raw = np.frombuffer(b"".join(bytes.fromhex(h) + b"\0" * pad for h in hex_hashes), dtype=np.uint8)
    return raw.reshape(len(hex_hashes), -1).view(np.uint64)


def find_near_duplicates(hex_hashes: list[str], max_hamming: int, chunk: int = 512) -> pd.DataFrame:
    """All pairs (i < j) whose Hamming distance is <= max_hamming."""
    packed = _pack(hex_hashes)
    pairs = []
    for start in range(0, len(packed), chunk):
        block = packed[start : start + chunk]
        dist = np.bitwise_count(block[:, None, :] ^ packed[None, :, :]).sum(axis=-1)
        rows, cols = np.nonzero(dist <= max_hamming)
        i = rows + start
        keep = cols > i
        pairs.append(np.stack([i[keep], cols[keep], dist[rows[keep], cols[keep]]], axis=1))
    out = np.concatenate(pairs) if pairs else np.empty((0, 3), dtype=int)
    return pd.DataFrame(out, columns=["i", "j", "hamming"]).astype(int)


def assign_groups(n: int, edges: list[tuple[int, int]]) -> list[str]:
    """Union-find over edges; group ids are numbered by first member, so they are deterministic."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    roots = [find(x) for x in range(n)]
    numbering = {root: k for k, root in enumerate(dict.fromkeys(roots))}
    return [f"g{numbering[r]:05d}" for r in roots]


def dedup(validation: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    dcfg = cfg["dedup"]
    df = validation[validation["usable"]].reset_index(drop=True)[
        ["image_id", "split", "label", "label_name", "path", "source_filename", "sha256"]
    ]

    fn = partial(compute_phash, hash_size=dcfg["hash_size"])
    with ProcessPoolExecutor(max_workers=dcfg.get("workers")) as pool:
        df["phash"] = list(pool.map(fn, df["path"], chunksize=32))
    df["patient_id"] = df["source_filename"].map(patient_id)

    near = find_near_duplicates(df["phash"].tolist(), dcfg["max_hamming"])
    edges = list(zip(near["i"], near["j"]))
    if dcfg["group_by_patient"]:
        for idx in df.dropna(subset=["patient_id"]).groupby("patient_id").groups.values():
            first = idx[0]
            edges += [(first, other) for other in idx[1:]]

    df["group_id"] = assign_groups(len(df), edges)
    df["group_size"] = df.groupby("group_id")["image_id"].transform("size")

    pairs = pd.DataFrame(
        {
            "image_id_a": df["image_id"].values[near["i"]],
            "image_id_b": df["image_id"].values[near["j"]],
            "split_a": df["split"].values[near["i"]],
            "split_b": df["split"].values[near["j"]],
            "label_a": df["label_name"].values[near["i"]],
            "label_b": df["label_name"].values[near["j"]],
            "hamming": near["hamming"].values,
            "exact": df["sha256"].values[near["i"]] == df["sha256"].values[near["j"]],
        }
    )
    return df.drop(columns=["sha256"]), pairs


def _split_pair(a: str, b: str) -> str:
    return "+".join(sorted((a, b)))


def summarize(df: pd.DataFrame, pairs: pd.DataFrame, cfg: dict) -> dict:
    groups = df.groupby("group_id")
    group_splits = groups["split"].agg(lambda s: "+".join(sorted(set(s))))
    multi_split = group_splits[group_splits.str.contains(r"\+")]

    patients = df.dropna(subset=["patient_id"]).groupby("patient_id")["split"].agg(
        lambda s: "+".join(sorted(set(s)))
    )
    shared_patients = patients[patients.str.contains(r"\+")]

    train_test_groups = multi_split[multi_split.str.contains("test") & multi_split.str.contains("train")]
    in_tt = df["group_id"].isin(train_test_groups.index)

    return {
        "n_images": len(df),
        "settings": cfg["dedup"],
        "near_duplicates": {
            "n_pairs": len(pairs),
            "n_exact_pairs": int(pairs["exact"].sum()),
            "n_cross_label_pairs": int((pairs["label_a"] != pairs["label_b"]).sum()),
            "pairs_by_split": {
                k: int(v)
                for k, v in pairs.apply(lambda r: _split_pair(r["split_a"], r["split_b"]), axis=1)
                .value_counts()
                .items()
            }
            if len(pairs)
            else {},
            "hamming_histogram": {int(k): int(v) for k, v in pairs["hamming"].value_counts().sort_index().items()},
        },
        "patients": {
            "n_with_id": int(df["patient_id"].notna().sum()),
            "n_patients": int(df["patient_id"].nunique()),
            "n_shared_across_splits": len(shared_patients),
            "shared_by_splits": {k: int(v) for k, v in shared_patients.value_counts().items()},
        },
        "groups": {
            "n_groups": int(df["group_id"].nunique()),
            "size_distribution": {
                int(k): int(v) for k, v in groups.size().value_counts().sort_index().items()
            },
            "n_spanning_splits": len(multi_split),
            "spanning_by_splits": {k: int(v) for k, v in multi_split.value_counts().items()},
            "train_images_grouped_with_test": int((in_tt & (df["split"] == "train")).sum()),
            "test_images_grouped_with_train": int((in_tt & (df["split"] == "test")).sum()),
        },
    }


def render_markdown(s: dict) -> str:
    nd, pt, gr = s["near_duplicates"], s["patients"], s["groups"]
    lines = [
        "# Duplicate / near-duplicate report",
        "",
        f"- Images: **{s['n_images']}**",
        (
            f"- pHash: `hash_size={s['settings']['hash_size']}` "
            f"({s['settings']['hash_size'] ** 2} bits), near-duplicate if Hamming "
            f"≤ `{s['settings']['max_hamming']}`; group by patient ID: `{s['settings']['group_by_patient']}`"
        ),
        "",
        "## Near-duplicate pairs (pHash)",
        "",
        (
            f"- Pairs: **{nd['n_pairs']}** (exact byte copies: {nd['n_exact_pairs']}, "
            f"different labels: {nd['n_cross_label_pairs']})"
        ),
        f"- By split: {nd['pairs_by_split']}",
        f"- Hamming histogram: {nd['hamming_histogram']}",
        "",
        "## Patient IDs (from filenames)",
        "",
        f"- Images with a patient ID: {pt['n_with_id']}, distinct patients: {pt['n_patients']}",
        (
            f"- Patient IDs appearing in more than one split: **{pt['n_shared_across_splits']}** "
            f"{pt['shared_by_splits']}"
        ),
        "",
        "## Groups",
        "",
        f"- Groups: **{gr['n_groups']}**",
        f"- Group size distribution (size: count): {gr['size_distribution']}",
        f"- Groups spanning more than one split: **{gr['n_spanning_splits']}** {gr['spanning_by_splits']}",
        (
            f"- Train images grouped with test: **{gr['train_images_grouped_with_test']}**; "
            f"test images grouped with train: **{gr['test_images_grouped_with_train']}**"
        ),
        "",
        (
            "Groups spanning train and test are potential leakage into the test set. "
            "Whether to drop those train images or only report them is recorded in `docs/decisions.md`."
        ),
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    interim_dir = Path(cfg["interim_dir"])
    validation = pd.read_csv(interim_dir / "validation.csv")
    df, pairs = dedup(validation, cfg)
    df.to_csv(interim_dir / "dedup.csv", index=False)
    pairs.to_csv(interim_dir / "near_duplicate_pairs.csv", index=False)

    summary = summarize(df, pairs, cfg)
    report_dir = Path(cfg["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / "dedup_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    (report_dir / "dedup_report.md").write_text(render_markdown(summary), encoding="utf-8")

    print(f"Images {summary['n_images']}, groups {summary['groups']['n_groups']}")
    print(f"Near-duplicate pairs: {summary['near_duplicates']['n_pairs']}")
    print(f"Groups spanning splits: {summary['groups']['spanning_by_splits']}")
    print(f"Report: {report_dir / 'dedup_report.md'}")


if __name__ == "__main__":
    main()
