import json

from cxr.monitoring.store import RequestLog

PREDICTION = {
    "prob_pneumonia": 0.8,
    "label": "PNEUMONIA",
    "quality": {
        "status": "warning",
        "flags": [
            {"stat": "noise", "value": 0.01, "side": "high", "limit": 0.006},
            {"stat": "blur", "value": 0.0, "side": "low", "limit": 0.001},
        ],
        "stats": {"noise": 0.01, "blur": 0.0},
    },
}


def test_appends_one_json_line_per_request(tmp_path):
    log = RequestLog(tmp_path / "sub" / "requests.jsonl")
    log.write("v1.0.0", PREDICTION)
    log.write("v1.0.0", PREDICTION)
    lines = (tmp_path / "sub" / "requests.jsonl").read_text().splitlines()
    assert len(lines) == 2
    record = json.loads(lines[0])
    assert record["model_version"] == "v1.0.0" and record["prob_pneumonia"] == 0.8
    assert record["quality_status"] == "warning" and record["flagged"] == ["blur", "noise"]
    assert record["stats"] == {"noise": 0.01, "blur": 0.0}


def test_none_path_writes_nothing(tmp_path):
    RequestLog(None).write("v1.0.0", PREDICTION)
    assert list(tmp_path.iterdir()) == []
