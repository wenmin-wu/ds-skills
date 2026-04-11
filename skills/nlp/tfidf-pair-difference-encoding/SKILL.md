---
name: nlp-tfidf-pair-difference-encoding
description: >
  Encodes text pairs by computing the absolute difference of their TF-IDF vectors, collapsing a pair into a single fixed-length feature vector.
---
# TF-IDF Pair Difference Encoding

## Overview

For text pair tasks (duplicate detection, semantic similarity, paraphrase identification), you need to represent two texts as a single feature vector. Fit TF-IDF on the combined corpus of both columns interleaved, then compute the absolute element-wise difference between each pair's vectors. This captures which terms differ between the two texts — shared terms cancel out, unique terms produce large values.

## Quick Start

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Interleave q1 and q2 into a single corpus
corpus = []
for _, row in df.iterrows():
    corpus.append(str(row["question1"]))
    corpus.append(str(row["question2"]))

tfidf = TfidfVectorizer(max_features=256)
vectors = tfidf.fit_transform(corpus)

# Absolute difference: q1 vectors are even indices, q2 are odd
X_diff = np.abs(vectors[0::2] - vectors[1::2])
```

## Workflow

1. Interleave both text columns into a single list (q1, q2, q1, q2, ...)
2. Fit `TfidfVectorizer` on the combined corpus
3. Transform to get sparse TF-IDF matrices
4. Slice even rows (q1) and odd rows (q2)
5. Compute `|q1 - q2|` element-wise as the pair encoding
6. Use as input features for a classifier (XGBoost, logistic regression)

## Key Decisions

- **max_features**: 128-512 is typical; higher captures more but increases dimensionality
- **Complement with**: Element-wise product `q1 * q2` captures co-occurrence, not just difference
- **Sparse output**: Result is sparse — feed directly to sklearn models or `.toarray()` for dense
- **Alternative**: Cosine similarity as a single scalar feature instead of full difference vector

## References

- [Quora EDA & Model selection](https://www.kaggle.com/code/philschmidt/quora-eda-model-selection-roc-pr-plots)
