"""Latency of /predict on a running API, one request at a time (D6, D13).

Sends every image of the split in splits.csv order, sequentially, to an already running server
(normally the Docker container). The first `n_warmup` requests are not counted. Latency is the
client-side wall time from sending the request to receiving the full response.

Writes reports/latency/<version>.json: p50, p95, mean, max (ms), machine info, pass/fail
against `max_p95_ms` in configs/api.yaml.

Usage:
  docker run --rm -d -p 8000:8000 --name cxr-api cxr-api:v1.0.0
  uv run python -m cxr.api.benchmark_latency --url http://localhost:8000
"""

import argparse
import json
import os
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from cxr.utils.config import load_config


def summarize_latency(latencies_ms: list[float], max_p95_ms: float) -> dict:
    """p50/p95 (linear interpolation, numpy default), mean, max; passed = p95 <= max_p95_ms."""
    lat = np.asarray(latencies_ms, dtype=np.float64)
    p95 = float(np.percentile(lat, 95))
    return {
        "n": len(lat),
        "p50_ms": float(np.percentile(lat, 50)),
        "p95_ms": p95,
        "mean_ms": float(lat.mean()),
        "max_ms": float(lat.max()),
        "acceptance": {"max_p95_ms": max_p95_ms, "value": p95, "passed": p95 <= max_p95_ms},
    }


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def time_request(client: httpx.Client, path: str) -> float:
    """Wall time in ms of one /predict call; raises on a non-2xx response."""
    with open(path, "rb") as f:
        data = f.read()  # read before the timer: disk I/O is not part of the server latency
    start = time.perf_counter()
    response = client.post("/predict", files={"file": (Path(path).name, data, "image/jpeg")})
    elapsed = (time.perf_counter() - start) * 1000
    response.raise_for_status()
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--split", default="test", choices=["test", "val"])
    parser.add_argument("--api-config", default="configs/api.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--out-dir", default="reports/latency")
    parser.add_argument("--target", default="docker cxr-api:v1.0.0", help="what is measured, for the report")
    args = parser.parse_args()

    bench_cfg = load_config(args.api_config)["benchmark"]
    data_cfg = load_config(args.data_config)
    splits = pd.read_csv(Path(data_cfg["processed_dir"]) / "splits.csv")
    paths = splits.loc[splits["split"] == args.split, "path"].tolist()

    with httpx.Client(base_url=args.url, timeout=30) as client:
        health = client.get("/health").json()
        n_warmup = bench_cfg["n_warmup"]
        for path in paths[:n_warmup]:
            time_request(client, path)
        latencies = [time_request(client, path) for path in paths]

    report = {
        "model_version": health["model_version"],
        "target": args.target,
        "url": args.url,
        "split": args.split,
        "n_warmup": n_warmup,
        "mode": "sequential, one image per request",
        **summarize_latency(latencies, bench_cfg["max_p95_ms"]),
        "machine": {"cpu": cpu_model(), "logical_cpus": os.cpu_count(), "platform": platform.platform()},
        "time": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{report['model_version']}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    a = report["acceptance"]
    print(f"{report['model_version']} via {args.url}: n={report['n']} (+{n_warmup} warm-up)")
    print(f"  p50 {report['p50_ms']:.1f} ms  p95 {report['p95_ms']:.1f} ms  max {report['max_ms']:.1f} ms")
    print(f"  p95 <= {a['max_p95_ms']} ms: {'PASS' if a['passed'] else 'FAIL'}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
