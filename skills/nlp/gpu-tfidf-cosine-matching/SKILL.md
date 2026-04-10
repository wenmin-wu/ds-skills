---
name: nlp-gpu-tfidf-cosine-matching
description: GPU-accelerated TF-IDF vectorization via RAPIDS cuML with chunked cosine similarity for large-scale text matching
domain: nlp
---

# GPU TF-IDF Cosine Matching

## Overview

For large-scale text matching (>100K documents), CPU-based TF-IDF + cosine similarity is too slow. Use RAPIDS cuML's TfidfVectorizer on GPU, convert sparse output to dense, then compute chunked cosine similarity with CuPy. Achieves 10-50x speedup over scikit-learn on text retrieval tasks.

## Quick Start

```python
import cudf
import cupy
from cuml.feature_extraction.text import TfidfVectorizer
from cuml.neighbors import NearestNeighbors

def gpu_tfidf_match(texts, threshold=0.7, max_features=25000, chunk_size=4096):
    """Find text matches via GPU TF-IDF + cosine similarity.
    
    Args:
        texts: cudf Series or list of strings
        threshold: cosine similarity cutoff
        max_features: TF-IDF vocabulary size
        chunk_size: rows per similarity batch
    Returns:
        list of matched index arrays per document
    """
    tfidf = TfidfVectorizer(stop_words='english', binary=True,
                            max_features=max_features)
    embeddings = tfidf.fit_transform(texts).toarray()  # dense on GPU
    
    N = len(texts)
    n_chunks = (N + chunk_size - 1) // chunk_size
    matches = [None] * N
    
    for j in range(n_chunks):
        a = j * chunk_size
        b = min(a + chunk_size, N)
        sims = cupy.matmul(embeddings, embeddings[a:b].T).T
        for k in range(b - a):
            idx = cupy.where(sims[k] > threshold)[0]
            matches[a + k] = cupy.asnumpy(idx)
    return matches

# Usage
df_cu = cudf.from_pandas(df)
text_matches = gpu_tfidf_match(df_cu['title'], threshold=0.7)
```

## Key Decisions

- **binary=True**: presence/absence often outperforms raw TF-IDF for short texts (titles)
- **max_features=25K**: limits vocabulary to reduce memory; increase for longer documents
- **Dense conversion**: cuML TF-IDF returns sparse; `.toarray()` needed for matmul
- **Threshold tuning**: 0.7 is a good start for product titles; lower for longer texts

## References

- Source: [part-2-rapids-tfidfvectorizer-cv-0-700](https://www.kaggle.com/code/cdeotte/part-2-rapids-tfidfvectorizer-cv-0-700)
- Competition: Shopee - Price Match Guarantee
