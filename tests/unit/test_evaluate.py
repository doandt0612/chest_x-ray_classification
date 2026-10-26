import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cxr.evaluation.evaluate import check_acceptance, check_splits_match, evaluate
from cxr.utils.provenance import file_sha256

BOOTSTRAP = {"n_boot": 50, "confidence": 0.95, "seed": 0}
ACCEPTANCE = {"auc": 0.93, "sensitivity": 0.90, "specificity": 0.75}


class LogitModel(nn.Module):
    """Input is a logit z; returns [0, z] so that P(PNEUMONIA) = sigmoid(z)."""

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))  # predict() reads the device from a parameter

    def forward(self, z):
        return torch.stack([torch.zeros_like(z), z], dim=1)


def _loader(probs, labels):
    z = torch.logit(torch.tensor(probs, dtype=torch.float64)).float()
    return DataLoader(TensorDataset(z, torch.tensor(labels)), batch_size=3)


def test_evaluate_uses_stored_threshold_without_rechoosing():
    # 4 NORMAL, 4 PNEUMONIA. At the stored threshold 0.5: NORMAL 0.6 is a false positive,
    # PNEUMONIA 0.4 is missed. Youden on this data would pick 0.7 instead, which must not happen.
    probs = [0.1, 0.2, 0.3, 0.6, 0.4, 0.7, 0.8, 0.9]
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    metadata = {"version": "v9.9.9", "threshold": 0.5}
    report, y, prob = evaluate(LogitModel(), metadata, _loader(probs, labels), BOOTSTRAP, "cpu")

    assert y.tolist() == labels and prob == pytest.approx(probs, abs=1e-6)
    m = report["metrics"]
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (3, 1, 1, 3)
    assert report["threshold"] == 0.5 and m["threshold"] == 0.5
    assert m["auc"] == pytest.approx(15 / 16)  # only (NORMAL 0.6, PNEUMONIA 0.4) is misordered
    assert report["ci"]["n_boot"] == 50
    acceptance = check_acceptance(m, ACCEPTANCE)
    assert acceptance["sensitivity"] == {"min": 0.90, "value": 0.75, "passed": False}
    assert acceptance["specificity"] == {"min": 0.75, "value": 0.75, "passed": True}  # inclusive


def _splits_and_metadata(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "splits.csv").write_text("image_id,split\na,test\n")
    data_cfg = {"processed_dir": str(processed)}
    metadata = {"provenance": {"splits_sha256": file_sha256(processed / "splits.csv")}}
    return data_cfg, metadata


def test_splits_match_passes_when_identical(tmp_path):
    data_cfg, metadata = _splits_and_metadata(tmp_path)
    check_splits_match(data_cfg, metadata)  # no error


def test_changed_splits_raise(tmp_path):
    data_cfg, metadata = _splits_and_metadata(tmp_path)
    (tmp_path / "processed" / "splits.csv").write_text("image_id,split\na,train\n")
    with pytest.raises(ValueError, match="splits.csv"):
        check_splits_match(data_cfg, metadata)
