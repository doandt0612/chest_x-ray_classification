# An MLOps Pipeline for Chest X-ray Image Classification

A reproducible MLOps pipeline for binary classification of chest X-ray images (NORMAL / PNEUMONIA): from public data → data validation → training → evaluation → a Docker-packaged inference API → input image quality monitoring.

> ⚠️ **Not for diagnostic use.** This is an academic project. The model is not a medical tool and must not be used clinically. The original data (Kermany et al., Mendeley Data) may only be used for research purposes.

## Data

- Source: [`hf-vision/chest-xray-pneumonia`](https://huggingface.co/datasets/hf-vision/chest-xray-pneumonia) (CC BY 4.0), commit pinned in `configs/data.yaml`.
- Original: Kermany et al. (2018), pediatric chest X-ray images (ages 1–5), Guangzhou Women and Children's Medical Center.

## Installation

<!-- TODO: fill in once the environment is finalized -->

## Re-running the full pipeline

<!-- TODO: 1–2 commands, e.g.: python scripts/run_pipeline.py --config configs/ -->

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
