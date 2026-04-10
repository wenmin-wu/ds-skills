---
name: tabular-recursive-feature-elimination
description: >
  Uses RFE with a tree estimator to iteratively remove least important features, selecting an optimal compact feature set.
---
# Recursive Feature Elimination (RFE)

## Overview

Start with all features, train a tree model, remove the least important feature(s), repeat until the desired count remains. RFE is more robust than single-pass importance ranking because feature importance changes as correlated features are removed. Use a fast estimator (DecisionTree) for the elimination loop, then train the final model (LightGBM) on selected features.

## Quick Start

```python
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier

def select_features_rfe(X_train, y_train, n_features=8):
    """Select top features via recursive elimination."""
    rfe = RFE(
        estimator=DecisionTreeClassifier(random_state=42),
        n_features_to_select=n_features,
        step=1,  # remove 1 feature per iteration
    )
    rfe.fit(X_train, y_train)

    selected = X_train.columns[rfe.support_].tolist()
    ranking = dict(zip(X_train.columns, rfe.ranking_))

    print(f"Selected {len(selected)} features: {selected}")
    return selected, ranking

# Usage
features, ranking = select_features_rfe(X_train, y_train, n_features=8)
model.fit(X_train[features], y_train)
```

## Workflow

1. Start with full feature set
2. Train tree estimator, rank features by importance
3. Remove lowest-importance feature(s)
4. Repeat until n_features remain
5. Train final model on selected features only

## Key Decisions

- **Estimator**: DecisionTree is fast; RandomForest is more stable but slower
- **step**: 1 for precise selection; higher for speed with many features
- **n_features**: Use RFECV (cross-validated) to find optimal count automatically
- **When to use**: 10-50 features where you suspect redundancy; not needed for <10 features
- **Alternative**: Permutation importance is model-agnostic and doesn't require retraining

## References

- Riiid Answer Correctness Prediction (Kaggle)
- Source: [riiid-answer-correctness-prediction-eda-modeling](https://www.kaggle.com/code/isaienkov/riiid-answer-correctness-prediction-eda-modeling)
