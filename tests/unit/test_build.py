import pytest
import torch

from cxr.models.build import build_model, check_normalization, count_parameters, set_train_mode

# pretrained=False keeps tests offline; the architecture is the same.
MODEL_CFG = {"backbone": "resnet18", "pretrained": False, "num_classes": 2, "freeze_backbone": False}
IMAGE_CFG = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


def test_output_is_two_class_logits():
    model = build_model(MODEL_CFG).eval()
    with torch.no_grad():
        out = model(torch.randn(4, 3, 224, 224))
    assert out.shape == (4, 2) and out.dtype == torch.float32


def test_unfrozen_model_trains_everything():
    counts = count_parameters(build_model(MODEL_CFG))
    assert counts["trainable"] == counts["total"]


def test_freeze_backbone_leaves_only_head_trainable():
    model = build_model({**MODEL_CFG, "freeze_backbone": True})
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"fc.weight", "fc.bias"}  # resnet18 head: Linear(512, 2)
    assert count_parameters(model)["trainable"] == 512 * 2 + 2


def test_frozen_backbone_gets_no_gradient():
    model = build_model({**MODEL_CFG, "freeze_backbone": True})
    model(torch.randn(2, 3, 64, 64)).sum().backward()
    assert model.conv1.weight.grad is None
    assert model.fc.weight.grad is not None


def test_normalization_check():
    model = build_model(MODEL_CFG)
    check_normalization(model, IMAGE_CFG)  # matches ImageNet: no error
    with pytest.raises(ValueError, match="image.mean"):
        check_normalization(model, {**IMAGE_CFG, "mean": [0.5, 0.5, 0.5]})
    with pytest.raises(ValueError, match="image.std"):
        check_normalization(model, {**IMAGE_CFG, "std": [0.5, 0.5, 0.5]})


def test_frozen_train_mode_keeps_batchnorm_stats_fixed():
    model = build_model({**MODEL_CFG, "freeze_backbone": True})
    before = model.bn1.running_mean.clone()
    set_train_mode(model, frozen_backbone=True)
    model(torch.randn(2, 3, 64, 64))
    assert torch.equal(model.bn1.running_mean, before)
    assert not model.training and model.fc.training


def test_unfrozen_train_mode_updates_batchnorm_stats():
    model = build_model(MODEL_CFG)
    before = model.bn1.running_mean.clone()
    set_train_mode(model, frozen_backbone=False)
    model(torch.randn(2, 3, 64, 64))
    assert not torch.equal(model.bn1.running_mean, before)
    assert model.training
