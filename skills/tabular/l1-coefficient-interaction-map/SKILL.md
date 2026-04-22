---
name: tabular-l1-coefficient-interaction-map
description: Extract and visualize per-subgroup feature coefficient signs from L1-regularized models as an interaction heatmap for EDA
---

# L1 Coefficient Interaction Map

## Overview

When data has a partitioning variable (group/category) with many values, understanding which features matter per group is hard. Train an L1-regularized model per group, extract coefficient signs (+1/0/-1), and assemble into a groups×features matrix. Visualized as a heatmap, this reveals: which features are universally important, which are group-specific, and whether feature interactions flip direction across groups.

## Quick Start

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

n_groups = df['group'].nunique()
n_features = len(feature_cols)
interactions = np.zeros((n_groups, n_features))

for i, g in enumerate(sorted(df['group'].unique())):
    subset = df[df['group'] == g]
    clf = LogisticRegression(solver='liblinear', penalty='l1', C=0.05)
    clf.fit(subset[feature_cols], subset['target'])
    interactions[i] = np.sign(clf.coef_[0])

plt.figure(figsize=(15, 8))
plt.matshow(interactions.T, fignum=1, aspect='auto', cmap='RdBu')
plt.xlabel('Group index')
plt.ylabel('Feature index')
plt.colorbar(label='Coefficient sign')
plt.title('Feature × Group Interaction Map')
plt.show()
```

## Workflow

1. Partition data by categorical variable
2. Fit L1-regularized LogisticRegression per partition (C=0.05 for aggressive sparsity)
3. Extract `clf.coef_[0]` → apply `np.sign()` to get +1/0/-1
4. Assemble into a (n_groups, n_features) matrix
5. Plot as heatmap with diverging colormap (RdBu)

## Key Decisions

- **L1 penalty**: produces sparse coefficients — zero means feature is irrelevant for that group
- **C value**: lower C = more sparsity. Use 0.01-0.1 for clear patterns
- **Sign vs magnitude**: signs show direction of effect; magnitude is less interpretable with regularization
- **Actionable insight**: columns with mixed signs across groups indicate strong interactions with the group variable

## References

- [Logistic Regression - [0.800]](https://www.kaggle.com/code/cdeotte/logistic-regression-0-800)
