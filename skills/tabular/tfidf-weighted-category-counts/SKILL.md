---
name: tabular-tfidf-weighted-category-counts
description: Convert per-group categorical event counts into TF-IDF-style features using log(1+tf/total) * log(N/df)
---

## Overview

Categorical count features (how often each activity / key / event type occurred per session) suffer from two problems: long sessions dominate raw counts, and common categories drown out rare-but-informative ones. Applying TF-IDF — the same transform used on text term counts — fixes both. Normalize each count by the session total (TF) and weight by how rare the category is across sessions (IDF). The result behaves like a dense numeric feature but preserves rarity signal. Fit IDF on train only, then reuse on test to avoid leakage.

## Quick Start

```python
import numpy as np

class CategoryTfidf:
    def __init__(self):
        self.idf = {}

    def fit_transform(self, counts_df):
        # counts_df: rows = session id, columns = categories, values = counts
        cnts = counts_df.sum(axis=1)
        out = counts_df.copy().astype(float)
        N = len(counts_df)
        for col in counts_df.columns:
            df_col = (counts_df[col] > 0).sum()
            idf = np.log(N / (df_col + 1))
            self.idf[col] = idf
            tf = counts_df[col] / cnts.replace(0, 1)
            out[col] = (1 + np.log1p(tf)) * idf
        return out

    def transform(self, counts_df):
        cnts = counts_df.sum(axis=1)
        out = counts_df.copy().astype(float)
        for col in counts_df.columns:
            tf = counts_df[col] / cnts.replace(0, 1)
            out[col] = (1 + np.log1p(tf)) * self.idf.get(col, 0)
        return out
```

## Workflow

1. Build a (session × category) count matrix — pivot with `crosstab` or `groupby().size().unstack()`
2. Fit TF-IDF on the training split: store per-category IDF
3. Transform both train and test using the stored IDF values
4. Concatenate with other tabular features and feed to LGBM / NN
5. Validate: on held-out OOF, TF-IDF typically beats raw counts by 0.3-1% in RMSE / AUC for long-tail categoricals

## Key Decisions

- **Fit IDF on train only**: fitting on train+test leaks the category distribution of test. Always split first.
- **+1 smoothing**: `log(N/(df+1))` avoids division-by-zero when a category appears only in test.
- **vs. raw counts**: raw counts are fine when the category vocabulary is small (<10); TF-IDF wins when >30 categories with long tail.
- **vs. sklearn TfidfVectorizer**: Vectorizer expects token lists; this applies the same math to pre-counted data directly.

## References

- [Silver Bullet | Single Model | 165 Features](https://www.kaggle.com/code/mcpenguin/silver-bullet-single-model-165-features)
