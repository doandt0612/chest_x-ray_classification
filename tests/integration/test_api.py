"""API tests on a tiny untrained model (no download, no models/ needed)."""

import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from fastapi.testclient import TestClient
from PIL import Image

from cxr.api.main import create_app
from cxr.api.schemas import DISCLAIMER
from cxr.models.build import build_model

IMAGE_CFG = {"size": 32, "resize": "squash", "mean": [0.5] * 3, "std": [0.5] * 3}
MODEL_CFG = {"backbone": "resnet18", "pretrained": False, "num_classes": 2, "freeze_backbone": False}
CHECKS = {
    "brightness": ["p0.5", "p99.5"],
    "contrast": ["p0.5", "p99.5"],
    "blur": ["p0.5", None],
    "noise": [None, "p99.5"],
    "blockiness": [None, "p99.5"],
}
WIDE = {"p0.5": -1e9, "p99.5": 1e9}  # nothing is ever flagged


def _write_setup(tmp_path, reference_stats=None, **api_overrides):
    torch.manual_seed(0)
    model_dir = tmp_path / "models" / "v0.0.1"
    model_dir.mkdir(parents=True)
    torch.save(build_model(MODEL_CFG).state_dict(), model_dir / "model.pt")
    metadata = {
        "version": "v0.0.1",
        "model": MODEL_CFG,
        "image": IMAGE_CFG,
        "label_names": {"0": "NORMAL", "1": "PNEUMONIA"},
        "threshold": 0.5,
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata))

    stats = reference_stats or {name: {"quantiles": WIDE} for name in CHECKS}
    reference = {"image": IMAGE_CFG, "stats": stats}
    (tmp_path / "stats.json").write_text(json.dumps(reference))
    monitor = {"reference": str(tmp_path / "stats.json"), "checks": CHECKS}
    (tmp_path / "monitor.yaml").write_text(yaml.safe_dump(monitor))

    cfg = {
        "model_dir": str(model_dir),
        "monitor_config": str(tmp_path / "monitor.yaml"),
        "max_upload_mb": 1,
        "allowed_formats": ["JPEG", "PNG"],
        "allowed_modes": ["L", "RGB"],
        "max_pixels": 4_000_000,
        "request_log": str(tmp_path / "logs" / "requests.jsonl"),
        **api_overrides,
    }
    return cfg


@pytest.fixture
def setup(tmp_path):
    cfg = _write_setup(tmp_path)
    return TestClient(create_app(cfg)), cfg


def _encode(img, fmt, **kw):
    buf = io.BytesIO()
    img.save(buf, fmt, **kw)
    return buf.getvalue()


def _xray(w=120, h=100, mode="L"):
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, (h, w), dtype=np.uint8)).convert(mode)


def _post(client, data, name="x.png"):
    return client.post("/predict", files={"file": (name, data, "application/octet-stream")})


def test_health(setup):
    client, _ = setup
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok", "model_version": "v0.0.1"}


@pytest.mark.parametrize("fmt,mode", [("PNG", "L"), ("JPEG", "L"), ("PNG", "RGB"), ("JPEG", "RGB")])
def test_predict_returns_full_response(setup, fmt, mode):
    client, _ = setup
    r = _post(client, _encode(_xray(mode=mode), fmt))
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["prob_pneumonia"] <= 1
    assert body["label"] == ("PNEUMONIA" if body["prob_pneumonia"] >= 0.5 else "NORMAL")
    assert body["threshold"] == 0.5 and body["model_version"] == "v0.0.1"
    assert body["disclaimer"] == DISCLAIMER
    assert body["quality"]["status"] == "ok" and body["quality"]["flags"] == []
    assert body["quality"]["stats"]["width"] == 120


def test_same_image_same_probability_for_gray_and_rgb_storage(setup):
    client, _ = setup
    img = _xray()
    p_gray = _post(client, _encode(img, "PNG")).json()["prob_pneumonia"]
    p_rgb = _post(client, _encode(img.convert("RGB"), "PNG")).json()["prob_pneumonia"]
    assert p_gray == p_rgb


def test_quality_warning_is_reported_but_prediction_still_returned(tmp_path):
    stats = {name: {"quantiles": WIDE} for name in CHECKS}
    stats["brightness"] = {"quantiles": {"p0.5": 0.9, "p99.5": 1.0}}  # every normal image is "too dark"
    client = TestClient(create_app(_write_setup(tmp_path, reference_stats=stats)))
    body = _post(client, _encode(_xray(), "PNG")).json()
    assert body["quality"]["status"] == "warning"
    assert [(f["stat"], f["side"], f["limit"]) for f in body["quality"]["flags"]] == [
        ("brightness", "low", 0.9)
    ]
    assert body["label"] in ("NORMAL", "PNEUMONIA")


def test_request_log_has_stats_but_no_image(setup):
    client, cfg = setup
    data = _encode(_xray(), "PNG")
    _post(client, data, name="patient_name.png")
    _post(client, data)
    with open(cfg["request_log"], encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]
    assert len(lines) == 2
    assert set(lines[0]) == {
        "time",
        "model_version",
        "prob_pneumonia",
        "label",
        "quality_status",
        "flagged",
        "stats",
    }
    assert "patient_name" not in Path(cfg["request_log"]).read_text(encoding="utf-8")


def test_request_log_off(tmp_path):
    cfg = _write_setup(tmp_path, request_log=None)
    assert _post(TestClient(create_app(cfg)), _encode(_xray(), "PNG")).status_code == 200
    assert not (tmp_path / "logs").exists()


@pytest.mark.parametrize(
    "data,status",
    [
        (b"", 400),
        (b"this is not an image", 400),
        (_encode(_xray(), "JPEG")[:300], 400),  # cut inside the header
        (_encode(_xray(), "JPEG")[:-2000], 400),  # header fine, pixel data cut
        (_encode(_xray(), "GIF"), 415),
        (_encode(_xray(mode="RGBA"), "PNG"), 415),
        (_encode(Image.fromarray(np.full((50, 50), 40000, dtype=np.uint16)), "PNG"), 415),
    ],
    ids=["empty", "text", "jpeg_cut_header", "jpeg_cut_pixels", "gif", "rgba_png", "png_16bit"],
)
def test_bad_input_gets_4xx(setup, data, status):
    client, _ = setup
    r = _post(client, data)
    assert r.status_code == status and r.json()["detail"]


def test_missing_file_field_is_422(setup):
    client, _ = setup
    assert client.post("/predict").status_code == 422


def test_upload_too_large_is_413(tmp_path):
    client = TestClient(create_app(_write_setup(tmp_path, max_upload_mb=0.001)))  # ~1 KB
    r = _post(client, _encode(_xray(400, 400), "PNG"))
    assert r.status_code == 413


def test_too_many_pixels_is_413(tmp_path):
    client = TestClient(create_app(_write_setup(tmp_path, max_pixels=100 * 100)))
    assert _post(client, _encode(_xray(120, 100), "PNG")).status_code == 413
    assert _post(client, _encode(_xray(100, 100), "PNG")).status_code == 200


def test_reference_with_other_preprocessing_fails_at_startup(tmp_path):
    cfg = _write_setup(tmp_path)
    ref = json.loads((tmp_path / "stats.json").read_text())
    ref["image"]["size"] = 224
    (tmp_path / "stats.json").write_text(json.dumps(ref))
    with pytest.raises(ValueError, match="size"):
        create_app(cfg)
