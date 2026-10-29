import hashlib
import os
import subprocess

import pytest

from cxr.utils.provenance import file_sha256, git_info, provenance


@pytest.fixture(autouse=True)
def _isolated_git(monkeypatch):
    """Inside a git hook GIT_DIR points at the real repo; global config may enforce signing or hooks."""
    for var in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@example.com")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def test_file_sha256_matches_hashlib(tmp_path):
    path = tmp_path / "f.bin"
    data = bytes(range(256)) * 5000  # > 1 MiB chunk
    path.write_bytes(data)
    assert file_sha256(path) == hashlib.sha256(data).hexdigest()


def test_clean_repo(repo):
    info = git_info(repo)
    assert len(info["commit"]) == 40 and info["dirty"] is False


def test_modified_tracked_file_is_dirty(repo):
    (repo / "src" / "a.py").write_text("x = 2\n")
    assert git_info(repo)["dirty"] is True


def test_untracked_file_dirty_only_under_code_dirs(repo):
    (repo / "notes.txt").write_text("scratch")
    assert git_info(repo)["dirty"] is False
    (repo / "src" / "new.py").write_text("y = 1\n")
    assert git_info(repo)["dirty"] is True


def test_outside_git_gives_none(tmp_path):
    assert git_info(tmp_path) == {"commit": None, "dirty": None}


def test_provenance_records_dataset_and_splits_hash(repo):
    (repo / "processed").mkdir()
    (repo / "processed" / "splits.csv").write_text("image_id,split\n")
    cfg = {"hf_repo": "org/ds", "revision": "abc", "processed_dir": "processed"}
    p = provenance(cfg, root=repo)
    assert p["dataset"] == {"hf_repo": "org/ds", "revision": "abc"}
    assert p["splits_sha256"] == hashlib.sha256(b"image_id,split\n").hexdigest()
    assert p["git"]["commit"] is not None
