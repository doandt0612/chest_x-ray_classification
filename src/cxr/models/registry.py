"""Versioned model store: promote a training run to models/<version>/.

models/<version>/model.pt       CPU state_dict of the best epoch
models/<version>/metadata.json  everything needed to rebuild, preprocess and decide:
                                model config, image preprocessing, label names, decision
                                threshold + val metrics, training config, provenance.

Usage: uv run python -m cxr.models.registry --run runs/<run_name> --version v1.0.0
"""

import argparse
import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path

import timm
import torch
from torch import nn

from cxr.models.build import build_model
from cxr.utils.provenance import provenance
from cxr.utils.seed import use_full_precision

VERSION_PATTERN = re.compile(r"^v\d+\.\d+\.\d+$")


def build_metadata(run_dir: Path, version: str) -> dict:
    with open(run_dir / "config.json", encoding="utf-8") as f:
        config = json.load(f)
    with open(run_dir / "val_metrics.json", encoding="utf-8") as f:
        val_metrics = json.load(f)
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu")

    # Runs record provenance at training time; older runs get it computed now, flagged as such.
    if "provenance" in config:
        prov, prov_source = config["provenance"], "training"
    else:
        prov, prov_source = provenance(config["data"]), "registration"

    return {
        "version": version,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_run": run_dir.name,
        "model": config["model"],
        "image": config["data"]["image"],
        "label_names": config["data"]["label_names"],
        "threshold": val_metrics["threshold"],
        "threshold_rule": val_metrics["threshold_rule"],
        "best_epoch": ckpt["epoch"],
        "val_metrics": val_metrics,
        "train": config["train"],
        "provenance": prov,
        "provenance_source": prov_source,
        "libraries": {"torch": torch.__version__, "timm": timm.__version__},
    }


def register(run_dir: str | Path, version: str, models_dir: str | Path = "models") -> Path:
    """Copy the run's best weights and metadata into models_dir/version; never overwrites."""
    if not VERSION_PATTERN.match(version):
        raise ValueError(f"Version must look like v1.0.0, got {version!r}")
    run_dir = Path(run_dir)
    out_dir = Path(models_dir) / version
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} already exists; pick a new version")

    metadata = build_metadata(run_dir, version)
    state = torch.load(run_dir / "best.pt", map_location="cpu")["model_state"]
    out_dir.mkdir(parents=True)
    try:
        torch.save(state, out_dir / "model.pt")
        with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
    except BaseException:
        shutil.rmtree(out_dir)  # a half-written version would block a retry under the same name
        raise
    return out_dir


def load_model(version_dir: str | Path, device: str = "cpu") -> tuple[nn.Module, dict]:
    """Rebuild the network (no download), load the weights, return (model in eval mode, metadata)."""
    use_full_precision()  # GPU inference must match the CPU API and the val threshold
    version_dir = Path(version_dir)
    with open(version_dir / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)
    model = build_model({**metadata["model"], "pretrained": False, "freeze_backbone": False})
    model.load_state_dict(torch.load(version_dir / "model.pt", map_location=device))
    return model.to(device).eval(), metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", required=True, help="training run directory, e.g. runs/<run_name>")
    parser.add_argument("--version", required=True, help="e.g. v1.0.0")
    parser.add_argument("--models-dir", default="models")
    args = parser.parse_args()

    out_dir = register(args.run, args.version, args.models_dir)
    with open(out_dir / "metadata.json", encoding="utf-8") as f:
        meta = json.load(f)
    vm = meta["val_metrics"]
    print(f"Registered {out_dir} from {meta['source_run']} (epoch {meta['best_epoch']})")
    print(
        f"  val AUC {vm['auc']:.4f}  threshold {meta['threshold']:.4f}  "
        f"sensitivity {vm['sensitivity']:.4f}  specificity {vm['specificity']:.4f}"
    )
    print(f"  git {meta['provenance']['git']}  (recorded at {meta['provenance_source']})")


if __name__ == "__main__":
    main()
