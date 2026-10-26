import json
import subprocess

import pytest
import torch

from cxr.models.build import build_model
from cxr.models.registry import load_model, register
from cxr.utils.provenance import file_sha256, git_info

MODEL_CFG = {"backbone": "resnet18", "pretrained": False, "num_classes": 2, "freeze_backbone": True}
DATA_CFG = {
    "hf_repo": "x/y",
    "revision": "abc",
    "processed_dir": "data/processed",
    "label_names": {"0": "NORMAL", "1": "PNEUMONIA"},
    "image": {"size": 32, "resize": "squash", "mean": [0.5] * 3, "std": [0.5] * 3},
}
VAL_METRICS = {
    "auc": 0.9,
    "threshold": 0.42,
    "sensitivity": 0.8,
    "specificity": 0.7,
    "threshold_rule": {"method": "youden"},
}
PROV = {
    "git": {"commit": "deadbeef", "dirty": False},
    "dataset": {"hf_repo": "x/y", "revision": "abc"},
    "splits_sha256": "00",
}


def _fake_run(tmp_path, with_provenance=True):
    """A run directory shaped like train.py output, with a GPU-free tiny checkpoint."""
    run = tmp_path / "runs" / "r1"
    run.mkdir(parents=True)
    torch.manual_seed(0)
    model = build_model(MODEL_CFG)
    torch.save(
        {"model_state": model.state_dict(), "model_cfg": MODEL_CFG, "epoch": 7, "val_auc": 0.9},
        run / "best.pt",
    )
    config = {"data": DATA_CFG, "model": MODEL_CFG, "train": {"seed": 42}, "eval": {}}
    if with_provenance:
        config["provenance"] = PROV
    (run / "config.json").write_text(json.dumps(config))
    (run / "val_metrics.json").write_text(json.dumps(VAL_METRICS))
    return run, model


def test_register_and_load_round_trip(tmp_path):
    run, model = _fake_run(tmp_path)
    out = register(run, "v1.0.0", tmp_path / "models")
    assert out == tmp_path / "models" / "v1.0.0"
    assert sorted(p.name for p in out.iterdir()) == ["metadata.json", "model.pt"]

    loaded, meta = load_model(out)
    assert not loaded.training
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        assert torch.allclose(loaded(x), model.eval()(x))
    # freeze_backbone from training is dropped at load time; it only matters for training.
    assert all(p.requires_grad for p in loaded.parameters())

    assert meta["version"] == "v1.0.0" and meta["source_run"] == "r1" and meta["best_epoch"] == 7
    assert meta["threshold"] == 0.42 and meta["val_metrics"]["auc"] == 0.9
    assert meta["image"] == DATA_CFG["image"] and meta["label_names"] == DATA_CFG["label_names"]
    assert meta["provenance"] == PROV and meta["provenance_source"] == "training"
    assert meta["train"] == {"seed": 42} and set(meta["libraries"]) == {"torch", "timm"}


def test_register_refuses_overwrite_and_bad_version(tmp_path):
    run, _ = _fake_run(tmp_path)
    register(run, "v1.0.0", tmp_path / "models")
    with pytest.raises(FileExistsError):
        register(run, "v1.0.0", tmp_path / "models")
    with pytest.raises(ValueError, match="v1.0.0"):
        register(run, "1.0", tmp_path / "models")


def test_failed_write_leaves_no_partial_version(tmp_path, monkeypatch):
    run, _ = _fake_run(tmp_path)

    def broken_dump(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("cxr.models.registry.json.dump", broken_dump)
    with pytest.raises(OSError):
        register(run, "v1.0.0", tmp_path / "models")
    assert not (tmp_path / "models" / "v1.0.0").exists()


def test_load_model_turns_off_tf32(tmp_path):
    run, _ = _fake_run(tmp_path)
    out = register(run, "v1.0.0", tmp_path / "models")
    torch.backends.cudnn.allow_tf32 = True
    load_model(out)
    assert not torch.backends.cudnn.allow_tf32


def test_old_run_gets_provenance_at_registration(tmp_path, monkeypatch):
    run, _ = _fake_run(tmp_path, with_provenance=False)
    splits = tmp_path / "data" / "processed" / "splits.csv"
    splits.parent.mkdir(parents=True)
    splits.write_text("image_id,split\na,train\n")
    monkeypatch.chdir(tmp_path)  # provenance() resolves processed_dir from the working directory
    meta = json.loads((register(run, "v0.1.0", "models") / "metadata.json").read_text())
    assert meta["provenance_source"] == "registration"
    assert meta["provenance"]["splits_sha256"] == file_sha256(splits)
    assert meta["provenance"]["dataset"] == {"hf_repo": "x/y", "revision": "abc"}


def test_git_info_outside_and_inside_repo(tmp_path):
    assert git_info(tmp_path) == {"commit": None, "dirty": None}
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "f.txt").write_text("a")
    env = ["-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(["git", *env, "add", "f.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", *env, "commit", "-qm", "x"], cwd=tmp_path, check=True)
    info = git_info(tmp_path)
    assert len(info["commit"]) == 40 and info["dirty"] is False
    (tmp_path / "notes.txt").write_text("x")  # untracked outside code dirs: still clean
    assert git_info(tmp_path)["dirty"] is False
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "new.py").write_text("x")  # untracked code: dirty
    assert git_info(tmp_path)["dirty"] is True
    (tmp_path / "src" / "new.py").unlink()
    (tmp_path / "f.txt").write_text("b")  # modified tracked file: dirty
    assert git_info(tmp_path)["dirty"] is True
