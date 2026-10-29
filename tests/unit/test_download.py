import hashlib
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cxr.data.download import count_images, extract_images


def _write_split(raw_dir, split, records, n_files=1):
    """records: [(bytes, source_filename, label)]; spread over n_files parquet shards."""
    (raw_dir / "data").mkdir(parents=True, exist_ok=True)
    chunk = -(-len(records) // n_files)
    for i in range(n_files):
        part = records[i * chunk : (i + 1) * chunk]
        table = pa.table(
            {
                "image": [{"bytes": b, "path": p} for b, p, _ in part],
                "label": [label for _, _, label in part],
            }
        )
        pq.write_table(table, raw_dir / "data" / f"{split}-{i:05d}-of-{n_files:05d}.parquet")


@pytest.fixture
def cfg(tmp_path):
    raw = tmp_path / "raw"
    _write_split(
        raw,
        "train",
        [(b"a", "IM-1.jpeg", 0), (b"bb", "person1_virus_1.JPEG", 1), (b"ccc", "x.png", 1)],
        n_files=2,
    )
    _write_split(raw, "validation", [(b"d", "IM-2.jpeg", 0)])
    _write_split(raw, "test", [(b"e", "noext", 1)])
    return {
        "raw_dir": str(raw),
        "images_dir": str(tmp_path / "images"),
        "label_names": {0: "NORMAL", 1: "PNEUMONIA"},
    }


def test_extract_writes_files_and_manifest(cfg):
    m = extract_images(cfg)
    assert list(m["image_id"]) == [
        "train_00000",
        "train_00001",
        "train_00002",
        "validation_00000",
        "test_00000",
    ]
    # ids run across parquet shards; extension lowercased; missing extension -> .jpeg
    assert [p.rsplit("/", 1)[1] for p in m["path"]] == [
        "train_00000.jpeg",
        "train_00001.jpeg",
        "train_00002.png",
        "validation_00000.jpeg",
        "test_00000.jpeg",
    ]
    assert list(m["label_name"]) == ["NORMAL", "PNEUMONIA", "PNEUMONIA", "NORMAL", "PNEUMONIA"]
    for row in m.itertuples():
        data = Path(row.path).read_bytes()
        assert hashlib.sha256(data).hexdigest() == row.sha256 and len(data) == row.n_bytes


def test_extract_rewrites_a_changed_file(cfg):
    m = extract_images(cfg)
    with open(m["path"][0], "wb") as f:
        f.write(b"tampered")
    m2 = extract_images(cfg)
    assert Path(m2["path"][0]).read_bytes() == b"a"


def test_missing_split_raises(cfg, tmp_path):
    for f in (tmp_path / "raw" / "data").glob("test-*.parquet"):
        f.unlink()
    with pytest.raises(FileNotFoundError, match="test"):
        extract_images(cfg)


def test_count_images_per_split_and_label():
    manifest = pd.DataFrame(
        {
            "split": ["train", "train", "train", "test"],
            "label_name": ["NORMAL", "PNEUMONIA", "PNEUMONIA", "NORMAL"],
        }
    )
    assert count_images(manifest) == {
        "test": {"NORMAL": 1, "PNEUMONIA": 0, "total": 1},
        "train": {"NORMAL": 1, "PNEUMONIA": 2, "total": 3},
    }
