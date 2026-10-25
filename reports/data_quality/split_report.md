# Split report

- `StratifiedGroupKFold`: target val fraction 0.15, seed 42; actual val share of the train pool: **14.30%**
- HF validation merged into train pool: `True`
- Excluded from train pool: {'too_small': 58, 'extreme_aspect': 26, 'rgb_mode': 283, 'flagged_total': 283, 'exact_duplicate': 26}
- Test is kept as published. No group_id spans more than one split (checked).

| Split | NORMAL | PNEUMONIA | Total | Groups | Pneumonia ratio | From source split |
| :-- | --: | --: | --: | --: | --: | :-- |
| train | 1155 | 3064 | 4219 | 2997 | 72.62% | {'train': 4205, 'validation': 14} |
| val | 193 | 511 | 704 | 501 | 72.59% | {'train': 702, 'validation': 2} |
| test | 234 | 390 | 624 | 427 | 62.50% | {'test': 624} |
