"""Load a registered model once and predict single images, with the input quality check.

Preprocessing is the model's own (metadata.json `image`) through the shared transforms.py,
and the probability is computed exactly as in train.predict, so API and offline evaluation
give the same number for the same image (D6: difference < 1e-5).
"""

from pathlib import Path

import torch
from PIL import Image

from cxr.data.transforms import build_transforms
from cxr.models.registry import load_model
from cxr.monitoring.drift import check_quality, check_reference_matches, load_bounds, load_reference
from cxr.monitoring.image_stats import image_stats


class Predictor:
    def __init__(self, model_dir: str | Path, monitor_cfg: dict):
        """monitor_cfg: configs/monitor.yaml (reference path + checks)."""
        self.model, self.metadata = load_model(model_dir, "cpu")
        self.image_cfg = self.metadata["image"]
        self.transform = build_transforms({"image": self.image_cfg}, train=False)
        reference = load_reference(monitor_cfg["reference"])
        check_reference_matches(reference, self.image_cfg)
        self.bounds = load_bounds(reference, monitor_cfg["checks"])

    @property
    def version(self) -> str:
        return self.metadata["version"]

    @torch.no_grad()
    def predict(self, img: Image.Image) -> dict:
        """{"label", "prob_pneumonia", "threshold", "quality": {"status", "flags", "stats"}}."""
        logits = self.model(self.transform(img).unsqueeze(0))
        prob = float(logits.softmax(dim=1)[0, 1].double())  # same op order as train.predict
        threshold = self.metadata["threshold"]
        stats = image_stats(img, self.image_cfg)
        return {
            "label": self.metadata["label_names"][str(int(prob >= threshold))],
            "prob_pneumonia": prob,
            "threshold": threshold,
            "quality": {**check_quality(stats, self.bounds), "stats": stats},
        }
