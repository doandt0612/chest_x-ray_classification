"""Input quality check: compare one image's stats with the train reference.

Bounds are quantiles of the train distribution (data/reference/stats.json), chosen per stat
in configs/monitor.yaml. A stat outside its bounds is flagged; any flag makes the status
"warning". The check never blocks a prediction, it only reports.

The reference must be computed with the same preprocessing (size, resize) as the image
stats, otherwise intensity, sharpness and noise are not comparable.
"""

import json
from pathlib import Path

Bounds = dict[str, tuple[float | None, float | None]]


def load_reference(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def check_reference_matches(reference: dict, image_cfg: dict) -> None:
    """Raise if the reference stats were computed with other preprocessing than image_cfg."""
    for key in ("size", "resize"):
        if reference["image"][key] != image_cfg[key]:
            raise ValueError(
                f"Reference stats use image.{key}={reference['image'][key]!r}, "
                f"but the model uses {image_cfg[key]!r}; recompute data/reference/stats.json"
            )


def load_bounds(reference: dict, checks: dict) -> Bounds:
    """{stat: (low, high)} from the reference quantiles; checks: the `checks` section of monitor.yaml."""
    bounds = {}
    for name, (low_q, high_q) in checks.items():
        quantiles = reference["stats"][name]["quantiles"]
        bounds[name] = (
            None if low_q is None else quantiles[low_q],
            None if high_q is None else quantiles[high_q],
        )
    return bounds


def check_quality(stats: dict[str, float], bounds: Bounds) -> dict:
    """{"status": "ok" | "warning", "flags": [{"stat", "value", "side", "limit"}, ...]}."""
    flags = []
    for name, (low, high) in bounds.items():
        value = stats[name]
        if low is not None and value < low:
            flags.append({"stat": name, "value": value, "side": "low", "limit": low})
        if high is not None and value > high:
            flags.append({"stat": name, "value": value, "side": "high", "limit": high})
    return {"status": "warning" if flags else "ok", "flags": flags}
