from pathlib import Path

import pytest

from cxr.utils.config import load_config


def test_load_config_parses_yaml(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("a: 1\nb: [x, null]\nnested:\n  c: 0.5  # comment\n", encoding="utf-8")
    assert load_config(path) == {"a": 1, "b": ["x", None], "nested": {"c": 0.5}}


def test_repo_configs_load():
    for name in ("data", "train", "eval", "monitor", "api"):
        assert isinstance(load_config(Path(__file__).parents[2] / "configs" / f"{name}.yaml"), dict)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")
