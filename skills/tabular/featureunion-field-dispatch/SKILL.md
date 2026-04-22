---
name: tabular-featureunion-field-dispatch
description: Use sklearn FeatureUnion with closure-based preprocessors to apply different vectorizers to different DataFrame columns in a single fit_transform call
---

# FeatureUnion Field Dispatch

## Overview

When a dataset has multiple text and categorical columns that each need different vectorization (TF-IDF for descriptions, CountVectorizer for names, token-pattern matching for categoricals), use a `FeatureUnion` with custom `preprocessor` closures. Each vectorizer receives a closure that extracts its target column from the row. This keeps the entire feature pipeline in one `fit_transform` call and produces a single sparse matrix.

## Quick Start

```python
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

default_preprocessor = CountVectorizer().build_preprocessor()

def build_preprocessor(field):
    idx = list(df.columns).index(field)
    return lambda x: default_preprocessor(x[idx])

vectorizer = FeatureUnion([
    ('name', CountVectorizer(
        ngram_range=(1, 2), max_features=50000,
        preprocessor=build_preprocessor('name'))),
    ('category', CountVectorizer(
        token_pattern='.+',
        preprocessor=build_preprocessor('category'))),
    ('brand', CountVectorizer(
        token_pattern='.+',
        preprocessor=build_preprocessor('brand'))),
    ('description', TfidfVectorizer(
        ngram_range=(1, 3), max_features=100000,
        preprocessor=build_preprocessor('description'))),
])

X = vectorizer.fit_transform(df.values)
```

## Workflow

1. Build a default preprocessor from `CountVectorizer().build_preprocessor()`
2. Create a closure factory that captures the column index and applies the preprocessor
3. Assign each vectorizer in the `FeatureUnion` its own closure
4. Use `token_pattern='.+'` for single-value categoricals (treat entire cell as one token)
5. Call `fit_transform(df.values)` — each row is a numpy array, closures extract the right column

## Key Decisions

- **df.values not df**: vectorizers expect string inputs; `.values` gives array rows to closures
- **token_pattern='.+'**: for categoricals, match the entire cell as a single token
- **max_features**: set per vectorizer to control total dimensionality
- **vs ColumnTransformer**: ColumnTransformer is cleaner for sklearn >=0.20, but FeatureUnion works on older versions and is more flexible with preprocessor closures

## References

- [ELI5 for Mercari](https://www.kaggle.com/code/lopuhin/eli5-for-mercari)
