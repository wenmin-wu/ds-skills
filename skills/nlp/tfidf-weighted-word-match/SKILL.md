---
name: nlp-tfidf-weighted-word-match
description: >
  Computes word overlap ratio between two texts weighted by inverse corpus frequency, giving rare shared words more importance than common ones.
---
# TF-IDF Weighted Word Match

## Overview

Simple word overlap (Jaccard) treats all words equally — "the" counts as much as "tensorflow". Weighting each shared word by inverse corpus frequency (IDF-style) makes rare shared terms contribute more to the similarity score. This produces a single float feature that strongly discriminates duplicate question pairs and similar text-matching tasks.

## Quick Start

```python
from collections import Counter
import numpy as np

# Build IDF-style weights from corpus
all_words = (" ".join(all_questions)).lower().split()
counts = Counter(all_words)
weights = {w: 1 / (c + 10000) for w, c in counts.items() if c >= 2}

stops = set(stopwords.words("english"))

def tfidf_word_match(row):
    q1 = {w for w in str(row["q1"]).lower().split() if w not in stops}
    q2 = {w for w in str(row["q2"]).lower().split() if w not in stops}
    if not q1 or not q2:
        return 0.0
    shared = [weights.get(w, 0) for w in q1 & q2]
    total = [weights.get(w, 0) for w in q1 | q2]
    return np.sum(shared) / (np.sum(total) + 1e-8)

df["tfidf_word_match"] = df.apply(tfidf_word_match, axis=1)
```

## Workflow

1. Concatenate all text into a single corpus, tokenize, count word frequencies
2. Compute IDF-style weight: `1 / (count + smoothing)` per word
3. For each text pair, find shared words (excluding stopwords)
4. Sum weights of shared words, divide by sum of all words' weights
5. Use as a feature for classification or ranking

## Key Decisions

- **Smoothing**: `eps=10000` prevents rare words from dominating; tune to corpus size
- **Min count**: Filter words appearing fewer than 2 times to avoid noise
- **Stopwords**: Remove standard stopwords before matching
- **Complement with**: Exact word match ratio (unweighted) as an additional feature

## References

- [Data Analysis & XGBoost Starter (0.35460 LB)](https://www.kaggle.com/code/anokas/data-analysis-xgboost-starter-0-35460-lb)
