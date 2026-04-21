---
name: tabular-tabnet-sklearn-wrapper
description: Wrap PyTorch TabNet in a scikit-learn BaseEstimator with built-in imputation and early stopping for use in VotingRegressor ensembles
---

# TabNet Sklearn Wrapper

## Overview

TabNet's native API is incompatible with scikit-learn's `VotingRegressor`/`VotingClassifier`. Wrapping it in a `BaseEstimator` with internal imputation, validation split, and early stopping lets you ensemble TabNet alongside LightGBM/XGBoost/CatBoost using standard sklearn patterns.

## Quick Start

```python
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from pytorch_tabnet.tab_model import TabNetRegressor

class TabNetWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, n_d=24, n_a=24, n_steps=3, lr=0.02,
                 patience=20, max_epochs=200, batch_size=1024):
        self.n_d = n_d; self.n_a = n_a; self.n_steps = n_steps
        self.lr = lr; self.patience = patience
        self.max_epochs = max_epochs; self.batch_size = batch_size
        self.imputer = SimpleImputer(strategy='median')

    def fit(self, X, y):
        X_imp = self.imputer.fit_transform(X)
        if hasattr(y, 'values'): y = y.values
        Xt, Xv, yt, yv = train_test_split(X_imp, y, test_size=0.2)
        self.model_ = TabNetRegressor(
            n_d=self.n_d, n_a=self.n_a, n_steps=self.n_steps,
            optimizer_params=dict(lr=self.lr))
        self.model_.fit(Xt, yt.reshape(-1,1),
            eval_set=[(Xv, yv.reshape(-1,1))],
            max_epochs=self.max_epochs, patience=self.patience,
            batch_size=self.batch_size, drop_last=False)
        return self

    def predict(self, X):
        return self.model_.predict(self.imputer.transform(X)).flatten()
```

## Workflow

1. Define `TabNetWrapper(BaseEstimator, RegressorMixin)` with TabNet hyperparams as `__init__` args
2. In `fit()`: impute → split for early stopping → train TabNet
3. In `predict()`: impute → predict → flatten
4. Use in `VotingRegressor(estimators=[('lgbm', lgbm), ('tabnet', TabNetWrapper())])`

## Key Decisions

- **Internal imputation**: TabNet cannot handle NaN — imputer must be inside the wrapper
- **Validation split**: 20% held out for early stopping; alternatively pass `eval_set` externally
- **All hyperparams in `__init__`**: required for sklearn `clone()` and `GridSearchCV` compatibility
- **`drop_last=False`**: prevents silent sample loss on small datasets

## References

- [LB0.494 with TabNet](https://www.kaggle.com/code/ichigoe/lb0-494-with-tabnet)
- [0.494 notebook](https://www.kaggle.com/code/cchangyyy/0-494-notebook)
