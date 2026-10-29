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

FastAPI app in `src/cxr/api/`, settings in `configs/api.yaml` (another file via `CXR_API_CONFIG`).
The registered model (`models/v1.0.0/`) is loaded once at startup.

```bash
uv run uvicorn cxr.api.main:create_app_from_env --factory --host 0.0.0.0 --port 8000
curl http://localhost:8000/health                       # {"status": "ok", "model_version": "v1.0.0"}
curl -F "file=@image.jpeg" http://localhost:8000/predict
```

`/predict` returns `label`, `prob_pneumonia`, the decision `threshold` (chosen on val), `model_version`,
a `quality` block and a "research use only" disclaimer. `quality` compares the image with the
train reference (`data/reference/stats.json`) on brightness, contrast, blur, noise and JPEG
blockiness (`configs/monitor.yaml`, decision D9); `status` is `warning` when any stat is out of
range. A warning never blocks the prediction.

Bad input gets a 4xx: 400 empty or unreadable image, 413 file over `max_upload_mb` or image over
`max_pixels`, 415 format other than JPEG/PNG or mode other than 8-bit gray/RGB, 422 no `file` field.
Images are never stored. Each request appends one JSON line of stats to `logs/requests.jsonl`
(`request_log: null` turns it off).

### Docker (CPU)

The image (`Dockerfile`) runs the same API on CPU. It bakes in `models/v1.0.0/`,
`data/reference/stats.json` and the two configs, so build it on a machine that has the
registered model (`models/` is not in git). Package versions come from `uv.lock`; torch and
torchvision use the same versions from the PyTorch CPU index instead of CUDA.

```bash
docker build -t cxr-api:v1.0.0 .
docker run --rm -p 8000:8000 cxr-api:v1.0.0
curl http://localhost:8000/health
curl -F "file=@image.jpeg" http://localhost:8000/predict
```

The container runs as a non-root user. The request log is written to `/app/logs/` inside the
container; mount it to keep it, e.g. `-v "$PWD/logs:/app/logs"` (the host folder must be
writable by the container user).

### Input monitoring evaluation

Checks the `quality` block on clean and synthetically degraded images (decision D10): blur, Gaussian
noise, darkening and JPEG compression, 4 levels each (`configs/monitor.yaml`, `degradations`;
`src/cxr/monitoring/degrade.py`). Every variant goes through the API's `Predictor`, so the check
and the probability are the same as `/predict`.

```bash
uv run python -m cxr.monitoring.evaluate_monitoring --version v1.0.0 --split val   # dry run, repeatable
uv run python -m cxr.monitoring.evaluate_monitoring --version v1.0.0               # test, once
```

Writes `reports/monitoring/v1.0.0/<split>_results.csv` (one row per image x variant),
`<split>_table.csv` (warning rate, flag rate per stat, AUC/sensitivity/specificity per kind x level)
and `<split>_summary.json` (on test: pass/fail against D6). Test results are never overwritten.

### Latency benchmark

Measures `/predict` on an already running server, sequentially, one test image per request
(decision D13; settings in `configs/api.yaml`, `benchmark`).

```bash
docker run --rm -d -p 8000:8000 --name cxr-api cxr-api:v1.0.0
uv run python -m cxr.api.benchmark_latency --url http://localhost:8000
docker stop cxr-api
```

Writes `reports/latency/v1.0.0.json` (p50, p95, mean, max in ms, machine info, pass/fail).

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

Input monitoring on test (D6, D10): 2.9% false alarms on clean images (≤ 5%, pass); 100% detected
at the strongest level of blur, noise, darkening and JPEG (≥ 90%, pass). Mild levels are a gap:
the model loses specificity before most images get a warning. Darkening ×0.7 drops specificity
from 0.735 to 0.440 with 32% of images warned; noise std 10 drops it to 0.615 with 40% warned;
darkening ×0.9 drops it to 0.645 with 4% warned. Sensitivity stays ≥ 0.99 at every level.
Full table: `reports/monitoring/v1.0.0/test_table.csv`.

Docker image: 2.10 GB as Docker Desktop shows it (D11). Size is reported only, not a requirement (D12).

Latency of `/predict` (D13): Docker container on CPU (AMD Ryzen 7 7840H, 16 threads, WSL2), one
request at a time, 624 test images after 20 warm-up requests: p50 41.9 ms, p95 84.7 ms, max 159.4 ms
(p95 ≤ 500 ms, pass). Report: `reports/latency/v1.0.0.json`.

## Model card

[docs/model_card.md](docs/model_card.md)

## References

[docs/references.md](docs/references.md)
