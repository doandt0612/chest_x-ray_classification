"""Mini end-to-end run of scripts/run_pipeline.py on ~50 synthetic images in a temp project.

Download needs the network, so the test writes the parquet shards itself and runs
extract_images (the offline half of stage 1); the pipeline then runs validate -> evaluate.
Training is 1 epoch on CPU without pretrained weights.
"""

import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from PIL import Image

from cxr.data.download import extract_images

REPO = Path(__file__).parents[2]
SCRIPT = REPO / "scripts" / "run_pipeline.py"
VERSION = "v0.0.1"


def _jpeg(rng) -> bytes:
    arr = rng.integers(0, 256, size=(300, 320), dtype=np.uint8)  # random noise: distinct pHash per image
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _write_parquet(raw_dir: Path, split: str, n_per_class: int, first_id: int, rng) -> None:
    """Filenames follow the source patterns, so dedup derives one patient per image."""
    records = []
    for i in range(first_id, first_id + n_per_class):
        records.append(({"bytes": _jpeg(rng), "path": f"IM-{i:04d}-0001.jpeg"}, 0))
        records.append(({"bytes": _jpeg(rng), "path": f"person{i}_bacteria_{i}.jpeg"}, 1))
    table = pa.table({"image": [r[0] for r in records], "label": [r[1] for r in records]})
    (raw_dir / "data").mkdir(parents=True, exist_ok=True)
    pq.write_table(table, raw_dir / "data" / f"{split}-00000-of-00001.parquet")


def _write_yaml(path: Path, cfg: dict) -> None:
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")


@pytest.fixture
def project(tmp_path, monkeypatch):
    """Temp project dir with configs/ and data/raw/images; all config paths are relative to it."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()

    data_cfg = yaml.safe_load((REPO / "configs" / "data.yaml").read_text(encoding="utf-8"))
    for section in ("validate", "dedup", "reference_stats"):
        data_cfg[section]["workers"] = 1
    _write_yaml(tmp_path / "configs" / "data.yaml", data_cfg)

    train_cfg = yaml.safe_load((REPO / "configs" / "train.yaml").read_text(encoding="utf-8"))
    train_cfg["model"]["pretrained"] = False  # offline
    train_cfg["train"].update(epochs=1, batch_size=8, num_workers=0, device="cpu")
    _write_yaml(tmp_path / "configs" / "train.yaml", train_cfg)

    eval_cfg = yaml.safe_load((REPO / "configs" / "eval.yaml").read_text(encoding="utf-8"))
    eval_cfg["bootstrap"]["n_boot"] = 20
    _write_yaml(tmp_path / "configs" / "eval.yaml", eval_cfg)

    rng = np.random.default_rng(0)
    raw_dir = Path(data_cfg["raw_dir"])
    _write_parquet(raw_dir, "train", 20, first_id=100, rng=rng)
    _write_parquet(raw_dir, "validation", 1, first_id=500, rng=rng)
    _write_parquet(raw_dir, "test", 6, first_id=1, rng=rng)
    manifest = extract_images(data_cfg)
    Path(data_cfg["interim_dir"]).mkdir(parents=True, exist_ok=True)
    manifest.to_csv(Path(data_cfg["interim_dir"]) / "manifest.csv", index=False)
    return tmp_path


def _run(project, *extra):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--version", VERSION, *extra],
        cwd=project,
        capture_output=True,
        text=True,
        check=False,
    )


def test_pipeline_runs_validate_to_evaluate(project):
    result = _run(project, "--from", "validate")
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    assert (
        "Pipeline finished: validate, dedup, split, reference_stats, train, register, evaluate"
        in result.stdout
    )

    splits = pd.read_csv(project / "data" / "processed" / "splits.csv")
    assert (splits["split"] == "test").sum() == 12  # test kept as published
    assert splits.groupby("group_id")["split"].nunique().max() == 1  # no group in two splits
    assert (project / "data" / "reference" / "stats.json").exists()
    assert (project / "runs" / f"pipeline_{VERSION}" / "best.pt").exists()
    assert (project / "models" / VERSION / "model.pt").exists()

    metrics = pd.read_json(project / "reports" / "evaluation" / VERSION / "test_metrics.json", typ="series")
    assert metrics["split"] == "test" and metrics["metrics"]["n"] == 12

    # A second run must refuse before running any stage (no overwrite of run, model or test results).
    again = _run(project, "--from", "validate")
    assert again.returncode != 0
    assert "Would overwrite" in again.stderr and "=== [1/" not in again.stdout


def test_failing_stage_stops_the_pipeline(project):
    (project / "data" / "interim" / "manifest.csv").unlink()  # validate needs it
    result = _run(project, "--from", "validate")
    assert result.returncode != 0
    assert "Stage validate failed" in result.stderr
    assert "=== [2/" not in result.stdout  # dedup never started


def test_guard_checks_only_selected_stages_and_version_format(tmp_path):
    """Empty dir, no configs: the guard must not read configs of stages that are not selected."""
    (tmp_path / "reports" / "evaluation" / VERSION).mkdir(parents=True)
    (tmp_path / "reports" / "evaluation" / VERSION / "test_metrics.json").write_text("{}")
    result = _run(tmp_path, "--from", "evaluate")
    assert result.returncode != 0 and "Would overwrite" in result.stderr

    bad = subprocess.run(
        [sys.executable, str(SCRIPT), "--version", "1.1", "--from", "register"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert bad.returncode != 0 and "Version must look like v1.0.0" in bad.stderr
    assert "=== [1/" not in bad.stdout
