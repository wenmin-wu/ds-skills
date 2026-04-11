---
name: tabular-multiclass-to-binary-collapse
description: >
  Trains on a finer-grained multiclass target (subtypes), then collapses non-baseline classes into a single positive class for binary submission.
---
# Multiclass-to-Binary Collapse

## Overview

When auxiliary labels provide finer-grained categories than the competition's binary target (e.g., disease subtypes A/B/C/D vs healthy/sick), training on the multiclass target gives the model more signal to learn decision boundaries. At prediction time, sum the probabilities of all non-baseline classes to get the binary positive probability. This typically improves binary classification by 0.5-2% because the model learns distinct patterns per subtype instead of lumping them together.

## Quick Start

```python
import numpy as np
from sklearn.model_selection import StratifiedKFold

# y_multi has subtypes: ['A', 'B', 'C', 'D', 'E']
# 'A' = healthy, 'B','C','D','E' = different conditions
# Binary target: class_0 = 'A', class_1 = everything else

# Train on multiclass target
model.fit(X_train, y_multi_train)

# Predict multiclass probabilities
probs = model.predict_proba(X_test)  # shape: (n, 5)

# Collapse: class_0 = P(A), class_1 = P(B) + P(C) + P(D) + P(E)
class_0_prob = probs[:, 0]  # baseline class
class_1_prob = probs[:, 1:].sum(axis=1)  # all non-baseline

submission['class_0'] = class_0_prob
submission['class_1'] = class_1_prob
```

## Workflow

1. Map auxiliary labels to a multiclass target (subtypes)
2. Train classifier on the multiclass target
3. Predict multiclass probabilities
4. Sum all non-baseline class probabilities for the positive class
5. Submit binary probabilities

## Key Decisions

- **Baseline class**: Must be index 0 in `model.classes_` — verify alignment
- **Oversampling**: Use the multiclass target for stratified oversampling (preserves subtype balance)
- **CV stratification**: Stratify folds by the multiclass target, not the binary
- **When to use**: Only when subtypes have genuinely different feature patterns

## References

- [ICR Identify Age](https://www.kaggle.com/code/vadimkamaev/icr-identify-age)
