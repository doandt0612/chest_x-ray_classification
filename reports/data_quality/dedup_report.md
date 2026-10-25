# Duplicate / near-duplicate report

- Images: **5856**
- pHash: `hash_size=16` (256 bits), near-duplicate if Hamming ≤ `20`; group by patient ID: `True`

## Near-duplicate pairs (pHash)

- Pairs: **34** (exact byte copies: 34, different labels: 0)
- By split: {'train+train': 28, 'test+test': 6}
- Hamming histogram: {0: 34}

## Patient IDs (from filenames)

- Images with a patient ID: 5856, distinct patients: 4097
- Patient IDs appearing in more than one split: **0** {}

## Groups

- Groups: **4096**
- Group size distribution (size: count): {1: 3369, 2: 328, 3: 179, 4: 67, 5: 57, 6: 39, 7: 23, 8: 14, 9: 7, 10: 4, 11: 4, 13: 1, 14: 2, 16: 1, 30: 1}
- Groups spanning more than one split: **0** {}
- Train images grouped with test: **0**; test images grouped with train: **0**

Groups spanning train and test are potential leakage into the test set. Whether to drop those train images or only report them is recorded in `docs/decisions.md`.
