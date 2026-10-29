"""FastAPI app: GET /health, POST /predict (multipart field "file").

The model is loaded once when the app is created. Bad input gets a clear 4xx:
400 unreadable or empty image, 413 file or image too large, 415 unsupported format or mode.
Images are never stored; only aggregate stats per request (store.py).

Usage:
  uv run uvicorn cxr.api.main:create_app_from_env --factory --host 0.0.0.0 --port 8000
  curl -F "file=@image.jpeg" http://localhost:8000/predict
"""

import io
import os

from fastapi import FastAPI, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from cxr.api.inference import Predictor
from cxr.api.schemas import Health, Prediction
from cxr.monitoring.store import RequestLog
from cxr.utils.config import load_config


def read_image(data: bytes, cfg: dict) -> Image.Image:
    """Decode upload bytes into a PIL image, or raise HTTPException with the matching 4xx."""
    if not data:
        raise HTTPException(400, "Empty file")
    try:
        img = Image.open(io.BytesIO(data))
    except Image.DecompressionBombError:
        raise HTTPException(413, "Image has too many pixels") from None
    except (UnidentifiedImageError, OSError):  # OSError: e.g. a file cut off inside the header
        raise HTTPException(400, "File is not a readable image") from None
    if img.format not in cfg["allowed_formats"]:
        raise HTTPException(415, f"Format {img.format} not supported; use one of {cfg['allowed_formats']}")
    if img.mode not in cfg["allowed_modes"]:
        raise HTTPException(415, f"Image mode {img.mode} not supported; use one of {cfg['allowed_modes']}")
    if img.width * img.height > cfg["max_pixels"]:
        raise HTTPException(413, f"Image has more than {cfg['max_pixels']} pixels")
    try:
        img.load()  # decode now, so a truncated file fails here and not inside the model
    except (OSError, SyntaxError):
        raise HTTPException(400, "Image data is corrupted or truncated") from None
    return img


def create_app(cfg: dict) -> FastAPI:
    """cfg: configs/api.yaml as a dict."""
    predictor = Predictor(cfg["model_dir"], load_config(cfg["monitor_config"]))
    request_log = RequestLog(cfg.get("request_log"))
    max_bytes = int(cfg["max_upload_mb"] * 1024 * 1024)

    app = FastAPI(title="Chest X-ray classification (research only)", version=predictor.version)

    @app.get("/health", response_model=Health)
    def health() -> Health:
        return Health(status="ok", model_version=predictor.version)

    # Plain def: FastAPI runs it in a worker thread, so CPU-bound inference does not block the server.
    @app.post("/predict", response_model=Prediction)
    def predict(file: UploadFile) -> Prediction:
        data = file.file.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise HTTPException(413, f"File larger than {cfg['max_upload_mb']} MB")
        result = predictor.predict(read_image(data, cfg))
        request_log.write(predictor.version, result)
        return Prediction(**result, model_version=predictor.version)

    return app


def create_app_from_env() -> FastAPI:
    """Entry point for uvicorn --factory; config file from CXR_API_CONFIG (default configs/api.yaml)."""
    return create_app(load_config(os.environ.get("CXR_API_CONFIG", "configs/api.yaml")))
