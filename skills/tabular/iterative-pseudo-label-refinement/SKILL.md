---
name: tabular-iterative-pseudo-label-refinement
description: Multi-round pseudo labeling with progressively confident test predictions merged into training plus OOF-based train label correction
---

# Iterative Pseudo-Label Refinement

## Overview

Standard pseudo labeling adds confident test predictions to training once. This technique iterates multiple rounds: each round retrains on the expanded dataset, produces better predictions, and raises the confidence bar. Additionally, it corrects noisy train labels using OOF predictions — if a train sample is predicted with extreme confidence opposite to its label, flip it. Each round typically gains 1-3% AUC.

## Quick Start

```python
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold

for itr in range(4):
    # Add confident test predictions as pseudo labels
    test['target'] = preds
    test.loc[test['target'] > 0.955, 'target'] = 1
    test.loc[test['target'] < 0.045, 'target'] = 0
    usable = test[(test['target'] == 1) | (test['target'] == 0)]
    new_train = pd.concat([train, usable]).reset_index(drop=True)

    # Correct noisy train labels using OOF
    new_train.loc[oof > 0.995, 'target'] = 1
    new_train.loc[oof < 0.005, 'target'] = 0

    # Retrain and collect new OOF + test predictions
    oof = np.zeros(len(train))
    preds = np.zeros(len(test))
    # ... model training loop ...
```

## Workflow

1. Train initial model, collect OOF predictions and test predictions
2. For each iteration (3-5 rounds):
   a. Threshold test predictions → add as pseudo-labeled training samples
   b. Correct train labels where OOF is extremely confident (> 0.995 or < 0.005)
   c. Retrain on expanded dataset, collect new OOF and test predictions
3. Use final-round predictions for submission

## Key Decisions

- **Threshold tightening**: start conservative (0.95/0.05), optionally tighten each round
- **Number of rounds**: 3-4 rounds usually saturate; more can introduce noise
- **OOF correction**: only flip at extreme confidence (> 0.995) to avoid corrupting clean labels
- **vs single-round**: iterative gains 2-5% over single-round pseudo labeling on noisy datasets

## References

- [Pseudo Labeling - QDA - [0.969]](https://www.kaggle.com/code/cdeotte/pseudo-labeling-qda-0-969)
