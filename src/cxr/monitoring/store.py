"""Per-request log for monitoring: aggregate numbers only, never the image or its file name.

One JSON line per request: time, model version, probability, label, quality status, flagged
stats and the image stats themselves. Enough to follow input drift over time offline.
"""

import json
import threading
from datetime import UTC, datetime
from pathlib import Path


class RequestLog:
    def __init__(self, path: str | Path | None):
        """path=None turns logging off."""
        self.path = None if path is None else Path(path)
        self._lock = threading.Lock()  # the API serves requests from several threads
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, model_version: str, prediction: dict) -> None:
        if self.path is None:
            return
        quality = prediction["quality"]
        record = {
            "time": datetime.now(UTC).isoformat(timespec="seconds"),
            "model_version": model_version,
            "prob_pneumonia": prediction["prob_pneumonia"],
            "label": prediction["label"],
            "quality_status": quality["status"],
            "flagged": sorted({f["stat"] for f in quality["flags"]}),
            "stats": quality["stats"],
        }
        line = json.dumps(record) + "\n"
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(line)
