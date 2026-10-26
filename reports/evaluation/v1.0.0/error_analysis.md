# Error analysis: v1.0.0 on test

Threshold 0.2313 (chosen on val, unchanged). Descriptive only: nothing here
changes the model or the threshold.

- Test outcomes: TN 172, FP 62, FN 2, TP 388. Test is kept as
  published, so 6 exact-duplicate NORMAL pairs are counted twice (D2).
- NORMAL above threshold: val 0.52%, test 26.50%
- Median prob of NORMAL images: val 0.0000, test 0.0175
- The 62 false positives come from 62 distinct patients: they are not clustered by patient

## Original image size and the label

Associations only. Size may go with the children's age or how the film was taken; the data
has no age field, so the cause cannot be separated.

AUC of one original size measure used alone as a score for PNEUMONIA (0.5 = no
information, below 0.5 = smaller value goes with PNEUMONIA). Train and val are filtered
(D3: too small, extreme aspect, RGB removed); test is not.

| Split | height | aspect_ratio |
| :-- | --: | --: |
| test | 0.080 | 0.704 |
| train | 0.086 | 0.860 |
| val | 0.107 | 0.851 |

On test NORMAL images, rank correlation of the predicted prob with original height -0.510,
with aspect ratio 0.428. By height third:

| Third | n | Height (px) | FP rate | Rank corr. prob vs aspect ratio |
| :-- | --: | :-- | --: | --: |
| small | 78 | 496 to 1113 | 57.7% | 0.406 |
| middle | 78 | 1122 to 1504 | 16.7% | -0.011 |
| large | 78 | 1505 to 2713 | 5.1% | 0.064 |

## False positives by predicted probability

| Band | Count |
| :-- | --: |
| [0.2313, 0.5) | 15 |
| [0.5, 0.9) | 15 |
| [0.9, 1] | 32 |

## Image stats: test false positives vs test true negatives

`auc` = P(FP > TN) + 0.5 P(tie) for a random FP image and a random TN image.

| Stat | Median FP | Median TN | auc |
| :-- | --: | --: | --: |
| width | 1504 | 1862 | 0.185 |
| height | 997 | 1426 | 0.165 |
| aspect_ratio | 1.533 | 1.285 | 0.781 |
| brightness | 0.4821 | 0.4878 | 0.500 |
| contrast | 0.2358 | 0.2375 | 0.483 |
| blur | 0.003136 | 0.002601 | 0.678 |
| dark_fraction | 0.09192 | 0.07591 | 0.604 |
| bright_fraction | 0.0001495 | 0 | 0.693 |

## Image stats: test NORMAL vs val NORMAL

`auc` = P(test > val) + 0.5 P(tie) for a random test image and a random val image.

| Stat | Median test | Median val | auc |
| :-- | --: | --: | --: |
| width | 1762 | 1588 | 0.634 |
| height | 1318 | 1298 | 0.491 |
| aspect_ratio | 1.353 | 1.219 | 0.704 |
| brightness | 0.4865 | 0.4775 | 0.552 |
| contrast | 0.2369 | 0.2385 | 0.462 |
| blur | 0.002677 | 0.003 | 0.365 |
| dark_fraction | 0.08101 | 0.09491 | 0.433 |
| bright_fraction | 0 | 0.0001594 | 0.341 |
