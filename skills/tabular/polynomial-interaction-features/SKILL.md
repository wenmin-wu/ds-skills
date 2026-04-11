---
name: tabular-polynomial-interaction-features
description: >
  Generates polynomial powers and interaction terms from selected numeric features to capture nonlinear relationships with the target.
---
# Polynomial Interaction Features

## Overview

Linear and tree models benefit from explicit polynomial and interaction features when key predictors have nonlinear relationships with the target. Sklearn's `PolynomialFeatures` generates all combinations up to degree N — including cross-terms (A×B, A×B²) that capture interactions the model might not find on its own. Select only the top 3–5 most predictive features to avoid combinatorial explosion.

## Quick Start

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer

# Select top correlated features only
key_features = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']

imputer = SimpleImputer(strategy='median')
X_train_poly = imputer.fit_transform(train[key_features])
X_test_poly = imputer.transform(test[key_features])

poly = PolynomialFeatures(degree=3, include_bias=False)
X_train_poly = poly.fit_transform(X_train_poly)
X_test_poly = poly.transform(X_test_poly)

poly_names = poly.get_feature_names_out(key_features)
train_poly = pd.DataFrame(X_train_poly, columns=poly_names, index=train.index)
test_poly = pd.DataFrame(X_test_poly, columns=poly_names, index=test.index)

# Check new features' correlation with target
correlations = train_poly.corrwith(train['TARGET']).abs().sort_values(ascending=False)
print(correlations.head(10))
```

## Workflow

1. Identify top 3–5 numeric features by target correlation
2. Impute missing values (median) before polynomial transform
3. Generate degree-N polynomial features (degree 2–3)
4. Check new feature correlations with target — some interactions outperform originals
5. Concatenate with main feature set; optionally prune low-correlation terms

## Key Decisions

- **Feature selection**: Only top predictors — 4 features at degree 3 = 34 new columns; 10 features = 285
- **Degree**: 2 is safe; 3 adds value if features are highly predictive; 4+ rarely helps
- **Imputation first**: PolynomialFeatures cannot handle NaN — impute before transforming
- **Pruning**: After generation, drop polynomial terms with near-zero target correlation

## References

- [Start Here: A Gentle Introduction](https://www.kaggle.com/code/willkoehrsen/start-here-a-gentle-introduction)
