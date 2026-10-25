import io

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from cxr.data.dedup import assign_groups, compute_phash, dedup, find_near_duplicates, patient_id

CFG = {"dedup": {"hash_size": 16, "max_hamming": 20, "group_by_patient": True, "workers": 1}}


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("person123_bacteria_45.jpeg", "person123_bacteria"),
        ("person123_virus_46.jpeg", "person123_virus"),
        ("person447_virus_921_1.jpeg", "person447_virus"),
        ("IM-0115-0001.jpeg", "IM-115"),
        ("IM-0429-0001-0002.jpeg", "IM-429"),
        ("NORMAL2-IM-0383-0001.jpeg", "NORMAL2-IM-383"),
        ("something_else.png", None),
    ],
)
def test_patient_id(filename, expected):
    assert patient_id(filename) == expected


def test_find_near_duplicates_threshold():
    hashes = ["0000", "0001", "00ff", "ffff"]  # distances from first: 1, 8, 16; 00ff-ffff: 8
    pairs = find_near_duplicates(hashes, max_hamming=1)
    assert pairs[["i", "j", "hamming"]].values.tolist() == [[0, 1, 1]]
    pairs = find_near_duplicates(hashes, max_hamming=7)
    assert sorted(map(tuple, pairs[["i", "j"]].values.tolist())) == [(0, 1), (1, 2)]
    pairs = find_near_duplicates(hashes, max_hamming=8)
    assert sorted(map(tuple, pairs[["i", "j"]].values.tolist())) == [(0, 1), (0, 2), (1, 2), (2, 3)]


def _flip_bits(hex_hash: str, n_bits: int, rng) -> str:
    bits = np.unpackbits(np.frombuffer(bytes.fromhex(hex_hash), dtype=np.uint8))
    bits[rng.choice(bits.size, n_bits, replace=False)] ^= 1
    return np.packbits(bits).tobytes().hex()


def test_find_near_duplicates_chunking_matches_full():
    # 10 clusters of 5 near-copies (0-6 flipped bits), shuffled so pairs cross chunk boundaries.
    rng = np.random.default_rng(0)
    bases = [rng.bytes(32).hex() for _ in range(10)]
    hashes = [_flip_bits(b, k, rng) for b in bases for k in (0, 1, 2, 4, 6)]
    hashes = [hashes[i] for i in rng.permutation(len(hashes))]

    full = find_near_duplicates(hashes, max_hamming=12)
    chunked = find_near_duplicates(hashes, max_hamming=12, chunk=7)
    pd.testing.assert_frame_equal(full, chunked)
    assert len(full) == 10 * 10  # every pair within a cluster (C(5,2) = 10), none across clusters
    assert (full["i"] // 7 != full["j"] // 7).any()


def test_find_near_duplicates_too_few_hashes():
    assert find_near_duplicates([], max_hamming=20).empty
    assert find_near_duplicates(["00ff"], max_hamming=20).empty


def test_assign_groups_is_transitive_and_deterministic():
    groups = assign_groups(5, [(3, 1), (1, 0)])
    assert groups[0] == groups[1] == groups[3]
    assert len({groups[0], groups[2], groups[4]}) == 3
    assert groups == ["g00000", "g00000", "g00001", "g00000", "g00002"]


def _xray_like(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (16, 16)).astype(np.uint8)
    return np.asarray(Image.fromarray(base).resize((320, 320), Image.BILINEAR))


def test_phash_survives_recompression_but_separates_different_images(tmp_path):
    original = tmp_path / "a.png"
    Image.fromarray(_xray_like(1)).save(original)
    buf = io.BytesIO()
    Image.fromarray(_xray_like(1)).resize((300, 300)).save(buf, format="JPEG", quality=70)
    recompressed = tmp_path / "a_small.jpeg"
    recompressed.write_bytes(buf.getvalue())
    other = tmp_path / "b.png"
    Image.fromarray(_xray_like(2)).save(other)

    hashes = [compute_phash(str(p), hash_size=16) for p in (original, recompressed, other)]
    pairs = find_near_duplicates(hashes, max_hamming=20)
    assert pairs[["i", "j"]].values.tolist() == [[0, 1]]


def test_dedup_groups_by_hash_and_patient(tmp_path):
    paths = []
    for name, seed in [("a", 1), ("a_copy", 1), ("b", 2), ("c", 3)]:
        p = tmp_path / f"{name}.png"
        Image.fromarray(_xray_like(seed)).save(p)
        paths.append(str(p))
    validation = pd.DataFrame(
        {
            "image_id": ["i0", "i1", "i2", "i3"],
            "split": ["train", "test", "train", "test"],
            "label": [1, 1, 1, 0],
            "label_name": ["PNEUMONIA", "PNEUMONIA", "PNEUMONIA", "NORMAL"],
            "path": paths,
            # i0/i1 are pixel copies; i2 shares patient 7 with i0; i3 is unrelated
            "source_filename": ["person7_virus_1.jpeg", "person9_virus_2.jpeg", "person7_virus_3.jpeg",
                                "IM-0001-0001.jpeg"],
            "sha256": ["x", "y", "z", "w"],
            "usable": [True, True, True, True],
        }
    )
    df, pairs = dedup(validation, CFG)
    assert df["group_id"].iloc[0] == df["group_id"].iloc[1] == df["group_id"].iloc[2]
    assert df["group_id"].iloc[3] != df["group_id"].iloc[0]
    assert df["group_size"].tolist() == [3, 3, 3, 1]
    assert pairs[["image_id_a", "image_id_b"]].values.tolist() == [["i0", "i1"]]


def test_dedup_skips_unusable_images(tmp_path):
    p = tmp_path / "a.png"
    Image.fromarray(_xray_like(1)).save(p)
    validation = pd.DataFrame(
        {
            "image_id": ["i0", "bad"],
            "split": ["train", "train"],
            "label": [0, 0],
            "label_name": ["NORMAL", "NORMAL"],
            "path": [str(p), "missing.png"],
            "source_filename": ["IM-0001-0001.jpeg", "IM-0002-0001.jpeg"],
            "sha256": ["x", "y"],
            "usable": [True, False],
        }
    )
    df, _ = dedup(validation, CFG)
    assert df["image_id"].tolist() == ["i0"]
