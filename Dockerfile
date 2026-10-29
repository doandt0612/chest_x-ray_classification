# CPU image for the inference API (src/cxr/api/). The registered model is baked in.
#   docker build -t cxr-api:v1.0.0 .
#   docker run --rm -p 8000:8000 cxr-api:v1.0.0
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY --from=ghcr.io/astral-sh/uv:0.11.18 /uv /usr/local/bin/uv

WORKDIR /app

# Dependencies first, so code changes do not reinstall them.
# Versions come from uv.lock. The lock holds CUDA torch for training, so torch/torchvision and the
# CUDA-only packages are dropped and the same torch versions are installed from the CPU index.
# CPU torch goes in first: otherwise timm pulls CUDA torch from PyPI, and the later CPU install
# is skipped because 2.14.1+cu130 already satisfies torch==2.14.1.
COPY pyproject.toml uv.lock ./
RUN uv pip install --system --no-cache --index-url https://download.pytorch.org/whl/cpu \
        torch==2.14.1 torchvision==0.29.1 \
    && uv export --frozen --no-dev --no-hashes --no-emit-project -o /tmp/req-lock.txt \
    && grep -vE '^(torch|torchvision|triton|nvidia-|cuda-)' /tmp/req-lock.txt > /tmp/req.txt \
    && uv pip install --system --no-cache -r /tmp/req.txt \
    && rm /tmp/req-lock.txt /tmp/req.txt

COPY src/ src/
RUN uv pip install --system --no-cache --no-deps .

# Only what the API reads at startup: its configs, the train reference stats, the model.
COPY configs/api.yaml configs/monitor.yaml configs/
COPY data/reference/stats.json data/reference/
COPY models/v1.0.0/ models/v1.0.0/

# Run as a non-root user; logs/ holds the per-request stats log (configs/api.yaml request_log).
RUN useradd --create-home app && mkdir logs && chown app logs
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "cxr.api.main:create_app_from_env", "--factory", "--host", "0.0.0.0", "--port", "8000"]
