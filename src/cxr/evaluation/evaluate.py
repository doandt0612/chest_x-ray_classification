"""Stage 7: evaluate a registered model on test with the threshold chosen on val.

Loads models/<version>/ (weights + metadata), predicts the split, and reports AUC,
sensitivity, specificity and the confusion counts with stratified bootstrap 95% CIs
(D7). On test, the point estimates are checked against the acceptance thresholds fixed
before looking at test (D6). The threshold comes from metadata.json; it is never re-chosen.

- Images are preprocessed with the model's own image config from metadata.json.
- splits.csv must have the hash recorded at training, otherwise the split is not the same.
- Test is one-shot: an existing test_metrics.json is never overwritten.
- The report records the git commit and the eval config used.

Writes reports/evaluation/<version>/<split>_metrics.json and <split>_predictions.csv.

Usage:
  uv run python -m cxr.evaluation.evaluate --version v1.0.0               # test, once
  uv run python -m cxr.evaluation.evaluate --version v1.0.0 --split val   # dry run, any number of times

The val dry run must match the run's val_predictions.csv: prob within 1e-4 (GPU at training
vs CPU now differ by ~1e-5). Confusion counts can differ by an image whose prob equals the
threshold, since Youden picks a val score.
"""

import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader

from cxr.data.dataset import build_dataset
from cxr.evaluation.bootstrap import CI_METRICS, bootstrap_ci
from cxr.evaluation.metrics import binary_metrics
from cxr.models.registry import load_model
from cxr.models.train import predict, resolve_device
from cxr.utils.config import load_config
from cxr.utils.provenance import file_sha256, provenance


def check_splits_match(data_cfg: dict, metadata: dict) -> None:
    """Raise if splits.csv differs from the one the model was trained with."""
    splits_hash = file_sha256(Path(data_cfg["processed_dir"]) / "splits.csv")
    if splits_hash != metadata["provenance"]["splits_sha256"]:
        raise ValueError("splits.csv differs from the one the model was trained with")


def check_acceptance(metrics: dict, acceptance: dict) -> dict:
    """{metric: {"min", "value", "passed"}}; a criterion passes when the point estimate >= min (D6)."""
    return {
        name: {"min": minimum, "value": metrics[name], "passed": bool(metrics[name] >= minimum)}
        for name, minimum in acceptance.items()
    }


def evaluate(model, metadata: dict, loader: DataLoader, bootstrap_cfg: dict, device) -> tuple[dict, ...]:
    """(report, labels, probs) at the model's stored threshold; labels and probs in loader order."""
    y, prob = predict(model, loader, device)
    threshold = metadata["threshold"]
    report = {
        "version": metadata["version"],
        "threshold": threshold,
        "metrics": binary_metrics(y, prob, threshold),
        "ci": bootstrap_ci(y, prob, threshold, bootstrap_cfg),
    }
    return report, y, prob


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--version", required=True, help="registered version, e.g. v1.0.0")
    parser.add_argument("--split", default="test", choices=["test", "val"])
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--out-dir", default="reports/evaluation")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) / args.version
    metrics_path = out_dir / f"{args.split}_metrics.json"
    if args.split == "test" and metrics_path.exists():
        raise FileExistsError(f"{metrics_path} exists; test is evaluated once per version")

    data_cfg, eval_cfg = load_config(args.data_config), load_config(args.eval_config)
    device = resolve_device(args.device)
    model, metadata = load_model(Path(args.models_dir) / args.version, device)
    check_splits_match(data_cfg, metadata)

    # The model's own preprocessing, so a later edit of data.yaml cannot silently change it.
    dataset = build_dataset({**data_cfg, "image": metadata["image"]}, args.split)
    loader = DataLoader(dataset, batch_size=64, num_workers=4)
    report, y, prob = evaluate(model, metadata, loader, eval_cfg["bootstrap"], device)
    if not (y == dataset.frame["label"].to_numpy()).all():
        raise RuntimeError("Prediction order does not match the dataset rows")

    report["split"] = args.split
    if args.split == "test":  # D6 criteria are defined on test only
        report["acceptance"] = check_acceptance(report["metrics"], eval_cfg["acceptance"])
    report["eval_config"] = eval_cfg
    report["provenance"] = provenance(data_cfg)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    dataset.frame[["image_id", "label"]].assign(prob=prob).to_csv(
        out_dir / f"{args.split}_predictions.csv", index=False
    )

    m = report["metrics"]
    print(f"{args.version} on {args.split} (n={m['n']}) at threshold {report['threshold']:.4f}")
    print(f"  tn {m['tn']}  fp {m['fp']}  fn {m['fn']}  tp {m['tp']}")
    for name in CI_METRICS:
        ci = report["ci"][name]
        line = f"  {name:<12} {ci['value']:.4f}  95% CI [{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]"
        if ci["ci_low"] == ci["ci_high"]:
            line += " (not estimable by bootstrap, D7)"
        if "acceptance" in report:
            a = report["acceptance"][name]
            line += f"  min {a['min']}  {'PASS' if a['passed'] else 'FAIL'}"
        print(line)
    print(f"Wrote {metrics_path} and {args.split}_predictions.csv")


if __name__ == "__main__":
    main()
