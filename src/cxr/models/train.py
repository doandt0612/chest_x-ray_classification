"""Stage 6: train the classifier on the train split, select it on val AUC.

- Cross-entropy with "balanced" class weights (train is ~73% PNEUMONIA).
- After each epoch, val AUC; the best epoch is saved, training stops after
  `patience` epochs without improvement.
- Test is never touched here.

Writes runs/<run_name>/best.pt (CPU state_dict + metadata), history.json (per-epoch
metrics), config.json (all configs as used + git commit and splits.csv hash), and for
the best epoch val_metrics.json (decision threshold chosen on val, sensitivity,
specificity) + val_predictions.csv.

Usage:
  uv run python -m cxr.models.train --train-config configs/train_baseline.yaml   # baseline
  uv run python -m cxr.models.train --train-config configs/train.yaml            # main model
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from cxr.data.dataset import build_dataset
from cxr.evaluation.metrics import binary_metrics, choose_threshold
from cxr.models.build import build_model, count_parameters, set_train_mode
from cxr.utils.config import load_config
from cxr.utils.provenance import provenance
from cxr.utils.seed import set_seed


def class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    """n_samples / (num_classes * n_c) per class, like sklearn's class_weight="balanced"."""
    counts = np.bincount(labels, minlength=num_classes)
    if (counts == 0).any():
        raise ValueError(f"Every class needs at least one training sample, got counts {counts.tolist()}")
    return torch.tensor(len(labels) / (num_classes * counts), dtype=torch.float32)


class EarlyStopping:
    """Track the best score; `step` returns True once `patience` epochs pass without a higher one."""

    def __init__(self, patience: int):
        self.patience = patience
        self.best = -float("inf")
        self.bad_epochs = 0

    def step(self, score: float) -> bool:
        if score > self.best:
            self.best = score
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def train_one_epoch(model, loader, loss_fn, optimizer, device, frozen_backbone: bool) -> float:
    """One pass over loader; returns the mean training loss per sample."""
    set_train_mode(model, frozen_backbone)
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
        n += len(y)
    return total / n


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """(labels, P(PNEUMONIA)) in loader order."""
    model.eval()
    labels, probs = [], []
    for x, y in loader:
        logits = model(x.to(device))
        # float64 so the values round-trip exactly through val_predictions.csv and the threshold.
        probs.append(logits.softmax(dim=1)[:, 1].double().cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(labels), np.concatenate(probs)


def fit(model, train_loader, val_loader, cfg: dict, out_dir: Path) -> list[dict]:
    """Train with early stopping on val AUC; saves the best epoch to out_dir/best.pt."""
    tcfg, mcfg = cfg["train"], cfg["model"]
    device = resolve_device(tcfg["device"])
    model.to(device)

    weights = class_weights(train_loader.dataset.labels, mcfg["num_classes"])
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=tcfg["lr"], weight_decay=tcfg["weight_decay"]
    )
    stopper = EarlyStopping(tcfg["patience"])

    history = []
    for epoch in range(1, tcfg["epochs"] + 1):
        start = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device, mcfg["freeze_backbone"])
        y_val, p_val = predict(model, val_loader, device)
        val_auc = float(roc_auc_score(y_val, p_val))

        improved = val_auc > stopper.best
        stop = stopper.step(val_auc)
        if improved:
            torch.save(
                {
                    # CPU tensors, so a CPU-only host (API, Docker) can load a GPU-trained checkpoint.
                    "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
                    "model_cfg": mcfg,
                    "epoch": epoch,
                    "val_auc": val_auc,
                },
                out_dir / "best.pt",
            )
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_auc": round(val_auc, 6),
            "best": improved,
            "seconds": round(time.perf_counter() - start, 1),
        }
        history.append(row)
        print(
            f"epoch {epoch:>3}  train_loss {train_loss:.4f}  val_auc {val_auc:.4f}"
            f"{'  *best' if improved else ''}  ({row['seconds']}s)"
        )
        if stop:
            print(f"Early stop: no val AUC gain for {tcfg['patience']} epochs")
            break
    return history


def make_loader(dataset, cfg: dict, shuffle: bool) -> DataLoader:
    tcfg = cfg["train"]
    return DataLoader(
        dataset,
        batch_size=tcfg["batch_size"],
        shuffle=shuffle,
        num_workers=tcfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
        persistent_workers=tcfg["num_workers"] > 0,
        generator=torch.Generator().manual_seed(tcfg["seed"]),  # fixes shuffle order and worker seeds
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--eval-config", default="configs/eval.yaml", help="decision threshold on val")
    parser.add_argument("--run-name", help="default: <timestamp>_<config file name>_seed<seed>")
    parser.add_argument("--epochs", type=int, help="override train.epochs, e.g. 1 for a quick check")
    parser.add_argument("--seed", type=int, help="override train.seed, e.g. for multi-seed runs")
    args = parser.parse_args()

    data_cfg = load_config(args.data_config)
    cfg = load_config(args.train_config)
    eval_cfg = load_config(args.eval_config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.seed is not None:
        cfg["train"]["seed"] = args.seed
    set_seed(cfg["train"]["seed"])

    seed = cfg["train"]["seed"]
    run_name = args.run_name or f"{datetime.now():%Y%m%d_%H%M%S}_{Path(args.train_config).stem}_seed{seed}"
    out_dir = Path(cfg["train"]["output_dir"]) / run_name
    out_dir.mkdir(parents=True, exist_ok=False)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        record = {"data": data_cfg, **cfg, "eval": eval_cfg, "provenance": provenance(data_cfg)}
        json.dump(record, f, indent=2)  # keys: data, model, train, eval, provenance

    train_ds = build_dataset(data_cfg, "train")
    val_ds = build_dataset(data_cfg, "val")
    model = build_model(cfg["model"], data_cfg["image"])
    print(f"Run {out_dir}  train {len(train_ds)}  val {len(val_ds)}  params {count_parameters(model)}")
    print(f"Class weights {class_weights(train_ds.labels, cfg['model']['num_classes']).tolist()}")

    val_loader = make_loader(val_ds, cfg, False)
    history = fit(model, make_loader(train_ds, cfg, True), val_loader, cfg, out_dir)
    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    best = max(history, key=lambda r: r["val_auc"])
    print(f"Best val AUC {best['val_auc']:.4f} at epoch {best['epoch']} -> {out_dir / 'best.pt'}")

    m = evaluate_val(model, val_loader, eval_cfg["threshold"], out_dir)
    print(
        f"Val at threshold {m['threshold']:.4f} ({eval_cfg['threshold']['method']}): "
        f"sensitivity {m['sensitivity']:.4f}  specificity {m['specificity']:.4f}"
    )


def evaluate_val(model, val_loader, threshold_cfg: dict, out_dir: Path) -> dict:
    """Reload best.pt, choose the decision threshold on val, write val_metrics.json + val_predictions.csv."""
    device = next(model.parameters()).device
    model.load_state_dict(torch.load(out_dir / "best.pt")["model_state"])
    y_val, p_val = predict(model, val_loader, device)

    metrics = binary_metrics(y_val, p_val, choose_threshold(y_val, p_val, threshold_cfg))
    metrics["threshold_rule"] = threshold_cfg
    with open(out_dir / "val_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    # Kept so the threshold can be re-chosen later without retraining.
    frame = val_loader.dataset.frame[["image_id", "label"]].assign(prob=p_val)
    frame.to_csv(out_dir / "val_predictions.csv", index=False)
    return metrics


if __name__ == "__main__":
    main()
