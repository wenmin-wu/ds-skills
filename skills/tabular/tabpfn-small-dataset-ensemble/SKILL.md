---
name: tabular-tabpfn-small-dataset-ensemble
description: >
  Ensembles TabPFN (a prior-fitted Bayesian transformer for small tabular data) with XGBoost, averaging probabilities for stronger predictions on datasets under 1000 rows.
---
# TabPFN Small Dataset Ensemble

## Overview

TabPFN is a transformer pre-trained on millions of synthetic tabular datasets — it performs Bayesian inference in a single forward pass without gradient-based training. On datasets with <1000 rows and <100 features, it matches or beats tuned XGBoost. But it has blind spots (feature interactions, specific distributions) where XGBoost excels. Ensembling both via probability averaging combines TabPFN's strong prior with XGBoost's flexibility, consistently outperforming either model alone on small-data competitions.

## Quick Start

```python
import numpy as np
import xgboost
from tabpfn import TabPFNClassifier
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator

class TabPFNXGBEnsemble(BaseEstimator):
    def __init__(self):
        self.imputer = SimpleImputer(strategy='median')
        self.classifiers = [
            xgboost.XGBClassifier(n_estimators=100, max_depth=3,
                                   learning_rate=0.2, subsample=0.9),
            TabPFNClassifier(N_ensemble_configurations=24),
            TabPFNClassifier(N_ensemble_configurations=64),
        ]

    def fit(self, X, y):
        X = self.imputer.fit_transform(X)
        unique, y_enc = np.unique(y, return_inverse=True)
        self.classes_ = unique
        for clf in self.classifiers:
            clf.fit(X, y_enc)
        return self

    def predict_proba(self, X):
        X = self.imputer.transform(X)
        probs = [clf.predict_proba(X) for clf in self.classifiers]
        return np.mean(probs, axis=0)

model = TabPFNXGBEnsemble()
model.fit(X_train, y_train)
preds = model.predict_proba(X_test)
```

## Workflow

1. Impute missing values (TabPFN doesn't handle NaN)
2. Fit TabPFN with multiple ensemble configurations (24, 64)
3. Fit XGBoost with standard hyperparameters
4. Average predicted probabilities across all classifiers
5. Use averaged probabilities for submission

## Key Decisions

- **N_ensemble_configurations**: Higher (64) = more accurate but slower; 24 is a good speed/accuracy tradeoff
- **TabPFN limits**: Max 1000 training rows, 100 features, 10 classes — filter features if needed
- **Imputation**: Must impute before TabPFN; median is safe for most tabular data
- **Weights**: Equal averaging is robust; weighted averaging rarely improves over equal

## References

- [Postprocessin_ Ensemble](https://www.kaggle.com/code/vadimkamaev/postprocessin-ensemble)
