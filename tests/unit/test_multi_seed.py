import json

import pytest
import torch

from cxr.evaluation.multi_seed import check_distinct_seeds, check_same_setup, load_run, summarize
from cxr.models.build import build_model

MODEL_CFG = {"backbone": "resnet18", "pretrained": False, "num_classes": 2, "freeze_backbone": False}


def _rows(aucs, sens=(0.9, 0.9, 0.9), spec=(0.7, 0.8, 0.9)):
    return [{"auc": a, "sensitivity": s, "specificity": p} for a, s, p in zip(aucs, sens, spec)]


def test_summarize_mean_and_sample_std():
    # AUC 0.95, 0.955, 0.96 -> mean 0.955, sample std 0.005; specificity 0.7, 0.8, 0.9 -> std 0.1
    s = summarize(_rows([0.95, 0.955, 0.96]), max_auc_std=0.01)
    assert s["n_runs"] == 3
    assert s["auc"]["mean"] == pytest.approx(0.955)
    assert s["auc"]["std"] == pytest.approx(0.005)
    assert s["sensitivity"]["std"] == pytest.approx(0.0)
    assert s["specificity"]["mean"] == pytest.approx(0.8)
    assert s["specificity"]["std"] == pytest.approx(0.1)  # ddof=1; population std would be 0.0816
    assert s["acceptance"]["auc_std"]["passed"] is True


def test_summarize_fails_when_auc_spread_too_large():
    s = summarize(_rows([0.90, 0.92, 0.94]), max_auc_std=0.01)  # sample std 0.02
    assert s["auc"]["std"] == pytest.approx(0.02)
    assert s["acceptance"]["auc_std"] == {"max": 0.01, "value": s["auc"]["std"], "passed": False}


@pytest.mark.parametrize("seeds", [[42], [42, 0, 42]])
def test_check_distinct_seeds_rejects_single_or_repeated(seeds):
    with pytest.raises(ValueError):
        check_distinct_seeds(seeds)


def test_check_distinct_seeds_accepts_distinct():
    check_distinct_seeds([42, 0, 1])


def _config(seed, lr=1e-4, eval_extra=None, commit="a"):
    return {
        "data": {"image": {"size": 224}},
        "model": {"backbone": "resnet18"},
        "train": {"seed": seed, "lr": lr},
        "eval": {"threshold": {"method": "youden"}, **(eval_extra or {})},
        "provenance": {"git": {"commit": commit}},
    }


def test_check_same_setup_ignores_seed_provenance_and_non_threshold_eval():
    configs = [_config(42), _config(0, eval_extra={"bootstrap": {"n_boot": 1000}}, commit="b")]
    check_same_setup(configs)


@pytest.mark.parametrize(
    "other",
    [
        _config(0, lr=1e-3),
        {**_config(0), "eval": {"threshold": {"method": "min_sensitivity"}}},
        {**_config(0), "model": {"backbone": "resnet50"}},
    ],
)
def test_check_same_setup_rejects_other_differences(other):
    with pytest.raises(ValueError, match="more than the seed"):
        check_same_setup([_config(42), other])


def test_load_run_restores_best_weights(tmp_path):
    torch.manual_seed(0)
    model = build_model(MODEL_CFG)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    torch.save(
        {"model_state": state, "model_cfg": MODEL_CFG, "epoch": 3, "val_auc": 0.9}, tmp_path / "best.pt"
    )
    (tmp_path / "config.json").write_text(json.dumps({"train": {"seed": 7}}))
    (tmp_path / "val_metrics.json").write_text(json.dumps({"auc": 0.9, "threshold": 0.4}))

    loaded, config, val_metrics = load_run(tmp_path, "cpu")
    assert not loaded.training
    assert config["train"]["seed"] == 7 and val_metrics["threshold"] == 0.4
    for k, v in loaded.state_dict().items():
        assert torch.equal(v, state[k]), k
