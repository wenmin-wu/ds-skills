---
name: nlp-two-stage-retrieve-rerank
description: Two-stage pipeline where an unsupervised bi-encoder retrieves KNN candidates and a supervised cross-encoder reranks them with sigmoid thresholding
---

# Two-Stage Retrieve-Rerank

## Overview

For large-scale matching tasks (content recommendation, question–document pairing), scoring every pair is infeasible. Stage 1 uses a fast bi-encoder to embed queries and documents separately, then KNN retrieves top-N candidates. Stage 2 passes each (query, candidate) pair through a cross-encoder for precise relevance scoring. Combines recall of dense retrieval with precision of cross-attention.

## Quick Start

```python
from cuml.neighbors import NearestNeighbors
import cupy as cp

# Stage 1: bi-encoder KNN retrieval
query_emb = encode(queries, bi_encoder)   # (Q, D)
doc_emb = encode(documents, bi_encoder)   # (N, D)
knn = NearestNeighbors(n_neighbors=50, metric='cosine')
knn.fit(cp.array(doc_emb))
indices = knn.kneighbors(cp.array(query_emb), return_distance=False)

# Stage 2: cross-encoder reranking
pairs = build_pairs(queries, documents, indices)  # Q*50 pairs
pairs['text'] = pairs['query_title'] + '[SEP]' + pairs['doc_title']
logits = cross_encoder_inference(pairs)
pairs['score'] = torch.sigmoid(logits).numpy()
matches = pairs[pairs['score'] > threshold]
```

## Workflow

1. Encode queries and documents independently with a sentence-transformer bi-encoder
2. GPU KNN (cuML or FAISS) retrieves top-N candidates per query (N=50–200)
3. Form all (query, candidate) pairs and concatenate with `[SEP]`
4. Score pairs with a cross-encoder (transformer + linear head → sigmoid)
5. Apply threshold to convert scores to binary matches
6. Group positive matches per query

## Key Decisions

- **N candidates**: 50–200 balances recall vs reranking cost; validate with recall@N on dev set
- **Bi-encoder model**: sentence-transformers (MiniLM, RoBERTa-base) for speed; fine-tune on domain data
- **Cross-encoder threshold**: very low (0.001–0.1) since cross-encoder is precise; tune on validation F1
- **GPU KNN**: cuML `NearestNeighbors` or FAISS for corpora > 100K documents

## References

- [LECR-stsb_roberta_base](https://www.kaggle.com/code/yuiwai/lecr-stsb-roberta-base)
- [LECR Inference P](https://www.kaggle.com/code/ragnar123/lecr-inference-p)
