# An MLOps Pipeline for Chest X-ray Image Classification

A reproducible MLOps pipeline for binary classification of chest X-ray images (NORMAL / PNEUMONIA): from public data → data validation → training → evaluation → a Docker-packaged inference API → input image quality monitoring.

> ⚠️ **Not for diagnostic use.** This is an academic project. The model is not a medical tool and must not be used clinically. The original data (Kermany et al., Mendeley Data) may only be used for research purposes.

## Data

- Source: [`hf-vision/chest-xray-pneumonia`](https://huggingface.co/datasets/hf-vision/chest-xray-pneumonia) (CC BY 4.0), commit pinned in `configs/data.yaml`.
- Original: Kermany et al. (2018), pediatric chest X-ray images (ages 1–5), Guangzhou Women and Children's Medical Center.

## Installation

Requires [uv](https://docs.astral.sh/uv/). Python 3.11 is pinned in `.python-version`; uv downloads it if needed.

```bash
uv sync    # creates .venv/ from pyproject.toml + uv.lock (torch/torchvision: CUDA 12.6 build)
```

Run every Python command through uv, e.g. `uv run pytest`.

## Re-running the full pipeline

Data pipeline (stages 1-5, settings in `configs/data.yaml`):

```bash
uv run python -m cxr.data.download         # fetch the pinned dataset revision, write manifest
uv run python -m cxr.data.validate         # image and label checks
uv run python -m cxr.data.dedup            # near-duplicates and patient groups
uv run python -m cxr.data.split            # train / val / test -> data/processed/splits.csv
uv run python -m cxr.data.reference_stats  # train-only reference stats -> data/reference/stats.json
```

Training (stage 6, settings in `configs/train_baseline.yaml` and `configs/train.yaml`):

```bash
uv run python -m cxr.models.train --train-config configs/train_baseline.yaml  # frozen backbone, head only
uv run python -m cxr.models.train --train-config configs/train.yaml           # fine-tune the whole network
```

Each run writes `runs/<timestamp>_<config>_seed<seed>/`: `best.pt` (best epoch on val AUC), `history.json`, `config.json`,
`val_metrics.json` (decision threshold chosen on val per `configs/eval.yaml`, sensitivity, specificity) and `val_predictions.csv`.
Options: `--seed N` (override the seed), `--epochs N` (e.g. 1 for a quick check), `--run-name NAME`.

Register the chosen run as a versioned model (never overwrites an existing version):

```bash
uv run python -m cxr.models.registry --run runs/<run_name> --version v1.0.0
```

This writes `models/v1.0.0/model.pt` (CPU weights) and `metadata.json` (model and preprocessing config,
decision threshold, val metrics, git commit, dataset revision, `splits.csv` hash).
GPU computations run in full float32 (TF32 off) so results match the CPU API.

<!-- TODO: evaluation stage; single entry point scripts/run_pipeline.py -->

## API

<!-- TODO: docker build / docker run, example calls to /health and /predict -->

## Directory structure

See [PLAN.md](../PLAN.md#1-cấu-trúc-thư-mục).

## Results

<!-- TODO: table of AUC, sensitivity, specificity (with bootstrap CI), p50/p95 latency, monitoring validation results -->

## Model card

[docs/model_card.md](docs/model_card.md)

## References

[docs/references.md](docs/references.md)
