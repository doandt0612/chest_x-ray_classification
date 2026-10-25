# Data validation report

- Images checked: **5856**, usable: **5856**
- Thresholds: `min_side=256`, `max_aspect_ratio=2.5`

## Label distribution

| Split | NORMAL | PNEUMONIA | Pneumonia ratio |
| :-- | --: | --: | --: |
| test | 234 | 390 | 62.50% |
| train | 1341 | 3875 | 74.29% |
| validation | 8 | 8 | 50.00% |

## Flags

| Split | corrupt | label_invalid | too_small | extreme_aspect | color | rgb_mode |
| :-- | --: | --: | --: | --: | --: | --: |
| test | 0 | 0 | 0 | 1 | 0 | 0 |
| train | 0 | 0 | 58 | 26 | 0 | 283 |
| validation | 0 | 0 | 0 | 0 | 0 | 0 |
| **total** | **0** | **0** | **58** | **27** | **0** | **283** |

`corrupt` and `label_invalid` make an image unusable; the other flags are warnings.

## Image properties

- Color modes: {'L': 5573, 'RGB': 283}
- Formats: {'JPEG': 5856}

| | min | p05 | median | p95 | max |
| :-- | --: | --: | --: | --: | --: |
| width | 384 | 848 | 1281 | 1966 | 2916 |
| height | 127 | 512 | 888 | 1697 | 2713 |
