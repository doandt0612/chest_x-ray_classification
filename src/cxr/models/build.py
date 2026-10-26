"""Build the classifier: an ImageNet-pretrained timm backbone with a new 2-class head.

The model takes the (3, size, size) tensors from transforms.py and returns logits of
shape (batch, num_classes). With `freeze_backbone`, only the head is trainable.
"""

import timm
from torch import nn


def build_model(model_cfg: dict, image_cfg: dict | None = None) -> nn.Module:
    """model_cfg: the `model` section of configs/train.yaml.

    image_cfg: the `image` section of configs/data.yaml. If given with a pretrained backbone,
    its mean/std must match the backbone's, otherwise inputs would be normalized wrongly.
    """
    model = timm.create_model(
        model_cfg["backbone"], pretrained=model_cfg["pretrained"], num_classes=model_cfg["num_classes"]
    )
    if model_cfg["pretrained"] and image_cfg is not None:
        check_normalization(model, image_cfg)
    if model_cfg["freeze_backbone"]:
        freeze_backbone(model)
    return model


def check_normalization(model: nn.Module, image_cfg: dict) -> None:
    pcfg = model.pretrained_cfg
    for key in ("mean", "std"):
        expected = [round(v, 4) for v in pcfg[key]]
        actual = [round(v, 4) for v in image_cfg[key]]
        if actual != expected:
            raise ValueError(
                f"image.{key} {actual} in configs/data.yaml does not match backbone "
                f"{pcfg.get('architecture')} ({expected})"
            )


def freeze_backbone(model: nn.Module) -> None:
    """Freeze every parameter except the classification head."""
    for p in model.parameters():
        p.requires_grad = False
    for p in model.get_classifier().parameters():
        p.requires_grad = True


def set_train_mode(model: nn.Module, frozen_backbone: bool) -> None:
    """Call instead of model.train() at the start of each training epoch.

    requires_grad=False does not stop BatchNorm from updating its running stats in train
    mode, nor Dropout from firing. A frozen backbone is kept in eval mode so the baseline
    really trains the head alone.
    """
    if frozen_backbone:
        model.eval()
        model.get_classifier().train()
    else:
        model.train()


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Total and trainable parameter counts, for logging."""
    return {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
