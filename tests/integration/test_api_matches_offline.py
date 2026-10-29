"""D6: API probability equals the offline evaluation for the same image (difference < 1e-5).

Needs the real model (models/v1.0.0, gitignored) and the test images; skipped otherwise.
"""

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from cxr.api.main import create_app
from cxr.utils.config import load_config

N_IMAGES = 40
PREDICTIONS = Path("reports/evaluation/v1.0.0/test_predictions.csv")
SPLITS = Path("data/processed/splits.csv")

pytestmark = pytest.mark.skipif(
    not (Path("models/v1.0.0/model.pt").exists() and SPLITS.exists()),
    reason="needs models/v1.0.0 and data/processed (not in git)",
)


def test_api_matches_offline_test_predictions():
    client = TestClient(create_app({**load_config("configs/api.yaml"), "request_log": None}))
    splits = pd.read_csv(SPLITS)
    test = splits[splits["split"] == "test"].reset_index(drop=True)
    offline = pd.read_csv(PREDICTIONS)
    assert (offline["image_id"] == test["image_id"]).all()

    for row, prob in zip(test.head(N_IMAGES).itertuples(), offline["prob"].head(N_IMAGES)):
        with open(row.path, "rb") as f:
            body = client.post("/predict", files={"file": ("x", f, "image/jpeg")}).json()
        assert abs(body["prob_pneumonia"] - prob) < 1e-5, row.image_id
