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

Evaluate a registered model on test, with the threshold chosen on val:

```bash
uv run python -m cxr.evaluation.evaluate --version v1.0.0
```

This writes `reports/evaluation/v1.0.0/test_metrics.json` (AUC, sensitivity, specificity,
confusion counts, bootstrap 95% CIs, pass/fail against the acceptance thresholds in
`configs/eval.yaml`), the git commit and `test_predictions.csv`. Test is evaluated once per
version: an existing `test_metrics.json` is never overwritten. `--split val` is a safe dry run;
its probabilities must match the run's `val_predictions.csv` within 1e-4.

Describe the errors on test (descriptive only, nothing is tuned):

```bash
uv run python -m cxr.evaluation.evaluate --version v1.0.0 --split val   # val predictions, needed below
uv run python -m cxr.evaluation.error_analysis --version v1.0.0
```

This writes `reports/evaluation/v1.0.0/error_analysis.{json,md}` and the ROC, confusion matrix and
probability histogram in `reports/figures/`.

Seed stability (decision D8): train the main model with more seeds, then score every run on test at
its own val threshold:

```bash
uv run python -m cxr.models.train --train-config configs/train.yaml --seed 0
uv run python -m cxr.models.train --train-config configs/train.yaml --seed 1
uv run python -m cxr.evaluation.multi_seed --runs runs/<seed42_run> runs/<seed0_run> runs/<seed1_run>
```

This checks first that the runs differ only by seed and share the same `splits.csv`, then writes
`reports/evaluation/multi_seed/per_run.csv` and `summary.json` (mean and sample std of AUC,
sensitivity, specificity; pass/fail on AUC std against `configs/eval.yaml`). Written once, never
overwritten. These runs measure spread only; v1.0.0 stays the selected model.

<!-- TODO: single entry point scripts/run_pipeline.py -->

## API

<!-- TODO: docker build / docker run, example calls to /health and /predict -->

## Directory structure

See [PLAN.md](../PLAN.md#1-cấu-trúc-thư-mục).

## Results

v1.0.0 on test (624 images, threshold 0.2313 chosen on val), with bootstrap 95% CIs:

| Metric | Test | Acceptance (D6) |
| :-- | :-- | :-- |
| AUC | 0.959 [0.941, 0.973] | ≥ 0.93, pass |
| Sensitivity | 0.995 [0.987, 1.000] | ≥ 0.90, pass |
| Specificity | 0.735 [0.675, 0.791] | ≥ 0.75, **fail** |

62 of 234 NORMAL test images are flagged as PNEUMONIA. See [docs/model_card.md](docs/model_card.md)
for the error analysis and limitations.

<!-- TODO: p50/p95 latency, monitoring validation results -->

## Model card

[docs/model_card.md](docs/model_card.md)

## References

[docs/references.md](docs/references.md)
