"""Record where a result came from: git commit of the code and a hash of the data split."""

import hashlib
import subprocess
from pathlib import Path

CODE_DIRS = ("src/", "configs/")


def git_info(repo: str | Path = ".") -> dict:
    """{"commit", "dirty"}; both None outside a git repository or without git.

    dirty = uncommitted changes to tracked files, or untracked files under CODE_DIRS
    (a new module imported by a run but never committed). Other untracked files are ignored.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    dirty = any(
        not line.startswith("??") or line[3:].startswith(CODE_DIRS) for line in status.splitlines() if line
    )
    return {"commit": commit, "dirty": dirty}


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def provenance(data_cfg: dict, root: str | Path = ".") -> dict:
    """Code commit + dataset revision + splits.csv hash, for config.json and model metadata."""
    return {
        "git": git_info(root),
        "dataset": {"hf_repo": data_cfg["hf_repo"], "revision": data_cfg["revision"]},
        "splits_sha256": file_sha256(Path(root) / data_cfg["processed_dir"] / "splits.csv"),
    }
