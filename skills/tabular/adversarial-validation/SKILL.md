---
name: tabular-adversarial-validation
description: >
  Trains a classifier to distinguish train from test data, detecting distribution shift and identifying leaked features.
---
# Adversarial Validation

## Overview

Train a binary classifier where train samples are label 0 and test samples are label 1. If the classifier achieves high AUC, there is significant distribution shift between train and test. The most important features in this classifier reveal which features are shifting — drop or transform them to reduce overfitting.

## Quick Start

```python
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import cross_val_score

def adversarial_validation(X_train, X_test, features):
    """Detect train/test distribution shift.

    Returns AUC score and feature importances.
    """
    X = np.vstack([X_train[features], X_test[features]])
    y = np.concatenate([np.zeros(len(X_train)), np.ones(len(X_test))])

    clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, n_jobs=-1)
    auc = cross_val_score(clf, X, y, cv=5, scoring='roc_auc').mean()

    clf.fit(X, y)
    importances = dict(zip(features, clf.feature_importances_))
    return auc, importances

auc, imp = adversarial_validation(train, test, feature_cols)
print(f"AUC: {auc:.3f}")  # >0.7 means significant shift
```

## Workflow

1. Combine train + test features (drop target column)
2. Assign labels: 0 = train, 1 = test
3. Train LightGBM classifier with 5-fold CV
4. If AUC > 0.7: inspect top features — they are shifting
5. Options: drop shifting features, apply distribution correction, or use as validation signal

## Key Decisions

- **AUC threshold**: ~0.5 = no shift, >0.7 = significant, >0.9 = likely temporal leak
- **Use as validation**: Sort train by predicted probability of being "test-like"; use the most test-like train samples as validation fold
- **Feature removal**: Only drop features if they are genuinely leaked (e.g., timestamps that only exist in test period)
- **Model choice**: LightGBM is fastest; any classifier works

## References

- 2019 Data Science Bowl (Kaggle)
- Source: [quick-and-dirty-regression](https://www.kaggle.com/code/artgor/quick-and-dirty-regression)
