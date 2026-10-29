import time

import httpx
import pytest

from cxr.api.benchmark_latency import summarize_latency, time_request


def test_summary_percentiles_and_acceptance():
    s = summarize_latency([float(v) for v in range(1, 101)], max_p95_ms=100)
    assert s["n"] == 100
    assert s["p50_ms"] == pytest.approx(50.5)  # linear interpolation between 50 and 51
    assert s["p95_ms"] == pytest.approx(95.05)
    assert s["mean_ms"] == 50.5 and s["max_ms"] == 100.0
    assert s["acceptance"]["passed"]
    # boundary is inclusive (all values equal: p95 is exact, no interpolation rounding)
    assert summarize_latency([500.0] * 10, max_p95_ms=500)["acceptance"]["passed"]
    assert not summarize_latency([600.0] * 10, max_p95_ms=500)["acceptance"]["passed"]


def _client(status, delay_s=0.0):
    def handler(request):
        time.sleep(delay_s)
        assert request.url.path == "/predict"
        assert b"image-bytes" in request.content  # the file is uploaded
        return httpx.Response(status, json={})

    return httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler))


def test_time_request_returns_ms(tmp_path):
    path = tmp_path / "x.jpeg"
    path.write_bytes(b"image-bytes")
    assert 40 <= time_request(_client(200, delay_s=0.05), str(path)) < 1000  # ms, not s


def test_time_request_raises_on_error(tmp_path):
    path = tmp_path / "x.jpeg"
    path.write_bytes(b"image-bytes")
    with pytest.raises(httpx.HTTPStatusError):
        time_request(_client(415), str(path))
