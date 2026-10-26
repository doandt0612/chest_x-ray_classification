import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from cxr.models.build import build_model
from cxr.models.train import EarlyStopping, class_weights, fit, predict
from cxr.utils.seed import set_seed


class ToyDataset(Dataset):
    """Label 1 images are brighter than label 0, so a few steps are enough to separate them."""

    def __init__(self, labels: list[int], seed: int):
        g = torch.Generator().manual_seed(seed)
        self.labels = labels
        self.x = torch.randn(len(labels), 3, 32, 32, generator=g) * 0.3 + torch.tensor(labels).view(
            -1, 1, 1, 1
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.x[i], self.labels[i]


CFG = {
    "model": {"backbone": "resnet18", "pretrained": False, "num_classes": 2, "freeze_backbone": False},
    "train": {"epochs": 4, "patience": 2, "lr": 1e-3, "weight_decay": 0.0, "device": "cpu"},
}


def test_class_weights_balanced():
    w = class_weights([0, 1, 1, 1], num_classes=2)
    assert torch.allclose(w, torch.tensor([4 / (2 * 1), 4 / (2 * 3)]))  # [2.0, 0.667]


def test_class_weights_missing_class_raises():
    with pytest.raises(ValueError, match="at least one"):
        class_weights([1, 1], num_classes=2)


def test_early_stopping_counts_epochs_without_gain():
    s = EarlyStopping(patience=2)
    # 0.7 and 0.8 improve; 0.8 (equal) and 0.75 do not -> stop on the second one.
    assert [s.step(v) for v in [0.7, 0.8, 0.8, 0.75]] == [False, False, False, True]
    assert s.best == 0.8


def test_early_stopping_resets_after_gain():
    s = EarlyStopping(patience=2)
    assert [s.step(v) for v in [0.7, 0.6, 0.9, 0.8, 0.85]] == [False, False, False, False, True]


def test_predict_keeps_order_and_returns_probabilities():
    ds = ToyDataset([0, 1, 0, 1, 1], seed=0)
    model = build_model(CFG["model"])
    labels, probs = predict(model, DataLoader(ds, batch_size=2), torch.device("cpu"))
    assert labels.tolist() == [0, 1, 0, 1, 1]
    assert probs.shape == (5,) and ((probs >= 0) & (probs <= 1)).all()


def test_fit_saves_best_epoch_and_learns(tmp_path):
    set_seed(0)
    train = ToyDataset([0, 1] * 16, seed=1)
    val = ToyDataset([0, 1] * 8, seed=2)
    model = build_model(CFG["model"])
    history = fit(
        model, DataLoader(train, batch_size=8, shuffle=True), DataLoader(val, batch_size=8), CFG, tmp_path
    )

    assert 1 <= len(history) <= CFG["train"]["epochs"]
    best = max(history, key=lambda r: r["val_auc"])
    assert best["val_auc"] > 0.9  # brightness separates the classes
    assert [r["epoch"] for r in history if r["best"]][-1] == best["epoch"]

    ckpt = torch.load(tmp_path / "best.pt")
    assert all(v.device.type == "cpu" for v in ckpt["model_state"].values())
    assert ckpt["epoch"] == best["epoch"] and np.isclose(ckpt["val_auc"], best["val_auc"], atol=1e-6)
    # The checkpoint loads into a freshly built model, as registry.py and the API will do.
    fresh = build_model(ckpt["model_cfg"])
    fresh.load_state_dict(ckpt["model_state"])
    _, p = predict(fresh, DataLoader(val, batch_size=8), torch.device("cpu"))
    assert np.isclose(roc_auc_score(val.labels, p), ckpt["val_auc"])


def test_fit_with_frozen_backbone_changes_only_head(tmp_path):
    set_seed(0)
    cfg = {**CFG, "model": {**CFG["model"], "freeze_backbone": True}, "train": {**CFG["train"], "epochs": 1}}
    model = build_model(cfg["model"])
    conv_before, fc_before = model.conv1.weight.clone(), model.fc.weight.clone()
    bn_before = model.bn1.running_mean.clone()
    fit(
        model,
        DataLoader(ToyDataset([0, 1] * 8, 1), batch_size=8),
        DataLoader(ToyDataset([0, 1] * 4, 2)),
        cfg,
        tmp_path,
    )
    assert torch.equal(model.conv1.weight, conv_before) and torch.equal(model.bn1.running_mean, bn_before)
    assert not torch.equal(model.fc.weight, fc_before)
