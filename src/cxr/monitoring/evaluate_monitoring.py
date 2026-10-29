"""Evaluate the input quality check on clean and synthetically degraded images (D6, D10).

Every image of the split is checked as is (clean) and after each degradation level in
configs/monitor.yaml. Each variant goes through the API's Predictor (same preprocessing,
quality check and probability as /predict). Reported per kind x level:
- warning_rate: share of images with a quality warning (false alarms on clean, detection otherwise)
- flag_<stat>: share of images flagged by each stat
- auc, sensitivity, specificity of the model at its stored threshold (why monitoring matters)

On test, the false alarm rate and the detection at the strongest level of each kind are
checked against `acceptance` in configs/monitor.yaml. Test is one-shot: existing results are
never overwritten. Val is a dry run and can be repeated.

Writes reports/monitoring/<version>/<split>_results.csv (one row per image x variant),
<split>_table.csv and <split>_summary.json.

Usage:
  uv run python -m cxr.monitoring.evaluate_monitoring --version v1.0.0 --split val   # dry run
  uv run python -m cxr.monitoring.evaluate_monitoring --version v1.0.0               # test, once
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from cxr.api.inference import Predictor
from cxr.evaluation.evaluate import check_splits_match
from cxr.evaluation.metrics import binary_metrics
from cxr.monitoring.degrade import KINDS, degrade
from cxr.utils.config import load_config
from cxr.utils.provenance import provenance

CLEAN = "clean"

_predictor: Predictor | None = None  # one per worker process


def variants(degradations: dict) -> list[tuple[str, float | None]]:
    """[("clean", None), (kind, level), ...] in config order; unknown kinds raise."""
    unknown = set(degradations) - set(KINDS)
    if unknown:
        raise ValueError(f"Unknown degradations {sorted(unknown)}; expected {KINDS}")
    return [(CLEAN, None)] + [(kind, level) for kind, levels in degradations.items() for level in levels]


def _init_worker(model_dir: str, monitor_cfg: dict) -> None:
    global _predictor
    torch.set_num_threads(1)  # parallelism comes from the worker processes
    _predictor = Predictor(model_dir, monitor_cfg)


def _one_image(task: tuple, monitor_cfg: dict) -> list[dict]:
    """All variants of one image. Noise is seeded by (noise_seed, row index): same result in any order."""
    index, image_id, path, label = task
    rng = np.random.default_rng([monitor_cfg["noise_seed"], index])
    rows = []
    with Image.open(path) as img:
        img.load()
        for kind, level in variants(monitor_cfg["degradations"]):
            x = img if kind == CLEAN else degrade(img, kind, level, rng)
            result = _predictor.predict(x)
            rows.append(
                {
                    "image_id": image_id,
                    "label": label,
                    "kind": kind,
                    "level": level,
                    "prob": result["prob_pneumonia"],
                    "warning": result["quality"]["status"] == "warning",
                    "flags": ";".join(f"{f['stat']}:{f['side']}" for f in result["quality"]["flags"]),
                }
            )
    return rows


def summarize(results: pd.DataFrame, monitor_cfg: dict, threshold: float) -> pd.DataFrame:
    """One row per variant: n, warning_rate, flag_<stat> rates, auc, sensitivity, specificity."""
    stats = list(monitor_cfg["checks"])
    table = []
    for kind, level in variants(monitor_cfg["degradations"]):
        sub = results[results["kind"] == kind]
        if level is not None:
            sub = sub[sub["level"] == level]
        flagged = sub["flags"].fillna("").str.split(";")
        m = binary_metrics(sub["label"], sub["prob"], threshold)
        table.append(
            {
                "kind": kind,
                "level": level,
                "n": len(sub),
                "warning_rate": float(sub["warning"].mean()),
                **{
                    f"flag_{s}": float(
                        flagged.map(lambda fs, s=s: any(f.startswith(f"{s}:") for f in fs)).mean()
                    )
                    for s in stats
                },
                "auc": m["auc"],
                "sensitivity": m["sensitivity"],
                "specificity": m["specificity"],
            }
        )
    return pd.DataFrame(table)


def check_acceptance(table: pd.DataFrame, monitor_cfg: dict) -> dict:
    """D6: false alarms on clean <= max_false_alarm; strongest level of each kind >= min_detection."""
    acc = monitor_cfg["acceptance"]
    clean = float(table.loc[table["kind"] == CLEAN, "warning_rate"].iloc[0])
    out = {
        "false_alarm": {
            "max": acc["max_false_alarm"],
            "value": clean,
            "passed": clean <= acc["max_false_alarm"],
        }
    }
    for kind, levels in monitor_cfg["degradations"].items():
        row = table[(table["kind"] == kind) & (table["level"] == levels[-1])]
        rate = float(row["warning_rate"].iloc[0])
        out[f"detection_{kind}"] = {
            "level": levels[-1],
            "min": acc["min_detection"],
            "value": rate,
            "passed": rate >= acc["min_detection"],
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--version", required=True, help="registered version, e.g. v1.0.0")
    parser.add_argument("--split", default="test", choices=["test", "val"])
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--monitor-config", default="configs/monitor.yaml")
    parser.add_argument("--out-dir", default="reports/monitoring")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    out_dir = Path(args.out_dir) / args.version
    summary_path = out_dir / f"{args.split}_summary.json"
    # Any earlier output counts, also from a run that crashed before writing the summary.
    if args.split == "test" and any(out_dir.glob("test_*")):
        raise FileExistsError(f"{out_dir} has test results; test is evaluated once per version")

    data_cfg, monitor_cfg = load_config(args.data_config), load_config(args.monitor_config)
    model_dir = str(Path(args.models_dir) / args.version)
    metadata = json.loads((Path(model_dir) / "metadata.json").read_text(encoding="utf-8"))
    check_splits_match(data_cfg, metadata)

    splits = pd.read_csv(Path(data_cfg["processed_dir"]) / "splits.csv")
    rows = splits[splits["split"] == args.split].reset_index(drop=True)
    tasks = list(zip(rows.index, rows["image_id"], rows["path"], rows["label"]))
    with ProcessPoolExecutor(
        args.workers, initializer=_init_worker, initargs=(model_dir, monitor_cfg)
    ) as pool:
        per_image = pool.map(_one_image, tasks, [monitor_cfg] * len(tasks), chunksize=4)
        results = pd.DataFrame([r for image_rows in per_image for r in image_rows])

    table = summarize(results, monitor_cfg, metadata["threshold"])
    summary = {
        "version": metadata["version"],
        "split": args.split,
        "n_images": len(rows),
        "threshold": metadata["threshold"],
        "table": json.loads(table.to_json(orient="records")),  # NaN level of clean -> null, valid JSON
        "monitor_config": monitor_cfg,
        "provenance": provenance(data_cfg),
    }
    if args.split == "test":  # D6 criteria are defined on test only
        summary["acceptance"] = check_acceptance(table, monitor_cfg)

    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / f"{args.split}_results.csv", index=False)
    table.to_csv(out_dir / f"{args.split}_table.csv", index=False)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    for name, a in summary.get("acceptance", {}).items():
        bound = f"<= {a['max']}" if "max" in a else f">= {a['min']}"
        print(f"  {name:<18} {a['value']:.4f}  {bound}  {'PASS' if a['passed'] else 'FAIL'}")
    print(f"Wrote {out_dir}/{args.split}_results.csv, _table.csv, _summary.json")


if __name__ == "__main__":
    main()
