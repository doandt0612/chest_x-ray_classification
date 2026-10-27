"""Seed stability (D6, D8): test metrics of several training runs, mean and standard deviation.

Each run is scored on test at its own decision threshold (val_metrics.json, chosen on val),
so every seed goes through the same procedure as v1.0.0. These numbers measure the spread
between seeds only; they are never used to pick a model (v1.0.0 stays selected on val).

- Runs must have distinct seeds, otherwise identical configs, and the splits.csv hash
  recorded at training. All of this is checked before any test prediction.
- Images are preprocessed with each run's own image config.
- The report is written once; an existing output directory is never overwritten.

Writes reports/evaluation/multi_seed/per_run.csv and summary.json.

Usage:
  uv run python -m cxr.evaluation.multi_seed --runs runs/<seed42> runs/<seed0> runs/<seed1>
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from cxr.data.dataset import build_dataset
from cxr.evaluation.evaluate import check_splits_match
from cxr.evaluation.metrics import binary_metrics
from cxr.models.build import build_model
from cxr.models.train import predict, resolve_device
from cxr.utils.config import load_config
from cxr.utils.provenance import provenance
from cxr.utils.seed import use_full_precision

SUMMARY_METRICS = ("auc", "sensitivity", "specificity")


def load_run(run_dir: str | Path, device) -> tuple[nn.Module, dict, dict]:
    """(model in eval mode with best.pt weights, config.json, val_metrics.json) of a training run."""
    use_full_precision()  # same numerics as the val threshold and evaluate.py
    run_dir = Path(run_dir)
    with open(run_dir / "config.json", encoding="utf-8") as f:
        config = json.load(f)
    with open(run_dir / "val_metrics.json", encoding="utf-8") as f:
        val_metrics = json.load(f)
    ckpt = torch.load(run_dir / "best.pt", map_location=device)
    model = build_model({**ckpt["model_cfg"], "pretrained": False, "freeze_backbone": False})
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), config, val_metrics


def check_distinct_seeds(seeds: list[int]) -> None:
    if len(seeds) < 2:
        raise ValueError(f"Need at least 2 runs to measure seed spread, got {len(seeds)}")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Runs must have distinct seeds, got {seeds}")


def check_same_setup(configs: list[dict]) -> None:
    """Raise unless the run configs (config.json) are identical apart from train.seed.

    Provenance is ignored, and of the eval section only `threshold` is compared: training
    uses nothing else from it, and later sections (bootstrap, acceptance) were added over time.
    """

    def without_seed(config: dict) -> dict:
        rest = {k: v for k, v in config.items() if k not in ("provenance", "eval")}
        return {
            **rest,
            "train": {k: v for k, v in config["train"].items() if k != "seed"},
            "eval_threshold": config["eval"]["threshold"],
        }

    first = without_seed(configs[0])
    for i, config in enumerate(configs[1:], start=1):
        other = without_seed(config)
        diff = sorted(k for k in first.keys() | other.keys() if first.get(k) != other.get(k))
        if diff:
            raise ValueError(f"Run {i} differs from run 0 in more than the seed: sections {diff}")


def summarize(rows: list[dict], max_auc_std: float) -> dict:
    """Mean and sample standard deviation (ddof=1) per metric, plus the D6 check on AUC std."""
    summary = {"n_runs": len(rows)}
    for name in SUMMARY_METRICS:
        values = np.array([r[name] for r in rows], dtype=np.float64)
        summary[name] = {"mean": float(values.mean()), "std": float(values.std(ddof=1))}
    auc_std = summary["auc"]["std"]
    summary["acceptance"] = {
        "auc_std": {"max": max_auc_std, "value": auc_std, "passed": bool(auc_std <= max_auc_std)}
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs", nargs="+", required=True, help="training run directories")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--out-dir", default="reports/evaluation/multi_seed")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists; the multi-seed test report is written once")

    data_cfg, eval_cfg = load_config(args.data_config), load_config(args.eval_config)
    device = resolve_device(args.device)

    # All checks before touching test, so a bad run list never costs a test pass.
    configs = []
    for run_dir in args.runs:
        with open(Path(run_dir) / "config.json", encoding="utf-8") as f:
            configs.append(json.load(f))
        check_splits_match(data_cfg, configs[-1])  # config.json has the same "provenance" key as metadata
    check_distinct_seeds([c["train"]["seed"] for c in configs])
    check_same_setup(configs)

    rows = []
    for run_dir in args.runs:
        model, config, val_metrics = load_run(run_dir, device)
        dataset = build_dataset({**data_cfg, "image": config["data"]["image"]}, "test")
        y, prob = predict(model, DataLoader(dataset, batch_size=64, num_workers=4), device)
        m = binary_metrics(y, prob, val_metrics["threshold"])
        rows.append(
            {
                "run": Path(run_dir).name,
                "seed": config["train"]["seed"],
                "val_auc": val_metrics["auc"],
                **{
                    k: m[k]
                    for k in ("threshold", "auc", "sensitivity", "specificity", "tn", "fp", "fn", "tp")
                },
            }
        )
        print(
            f"{rows[-1]['run']}  seed {rows[-1]['seed']}  test auc {m['auc']:.4f}  "
            f"sensitivity {m['sensitivity']:.4f}  specificity {m['specificity']:.4f}"
        )

    summary = summarize(rows, eval_cfg["multi_seed"]["max_auc_std"])
    report = {
        "split": "test",
        "runs": rows,
        "summary": summary,
        "eval_config": eval_cfg,
        "provenance": provenance(data_cfg),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "per_run.csv", index=False)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    for name in SUMMARY_METRICS:
        print(f"  {name:<12} {summary[name]['mean']:.4f} ± {summary[name]['std']:.4f}")
    a = summary["acceptance"]["auc_std"]
    print(f"  AUC std {a['value']:.4f}  max {a['max']}  {'PASS' if a['passed'] else 'FAIL'}")
    print(f"Wrote {out_dir / 'per_run.csv'} and summary.json")


if __name__ == "__main__":
    main()
