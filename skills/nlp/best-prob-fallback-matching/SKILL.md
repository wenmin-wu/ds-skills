---
name: nlp-best-prob-fallback-matching
description: When no candidate passes the threshold for a query, fall back to the single highest-scoring match to guarantee at least one prediction per query
---

# Best-Prob Fallback Matching

## Overview

In retrieval/matching tasks scored by MAP or recall, returning zero predictions for a query is maximally penalized. When a reranker's sigmoid threshold filters out all candidates for some queries, a fallback ensures every query gets at least one prediction — the highest-scoring candidate, even if below threshold. This is especially important with aggressive (low) thresholds where a few edge-case queries still end up empty.

## Quick Start

```python
import pandas as pd

def apply_with_fallback(df, threshold, query_col, doc_col, score_col):
    df = df.sort_values(score_col, ascending=False)
    # Queries with at least one match above threshold
    pos = df[df[score_col] > threshold]
    matched = pos.groupby(query_col)[doc_col].agg(list).reset_index()
    matched_ids = set(matched[query_col])
    # Queries with no match: take top-1 candidate
    remaining = df[~df[query_col].isin(matched_ids)]
    fallback = remaining.groupby(query_col).head(1)[[query_col, doc_col]]
    fallback[doc_col] = fallback[doc_col].apply(lambda x: [x])
    return pd.concat([matched, fallback], ignore_index=True)

result = apply_with_fallback(pairs, threshold=0.001,
    query_col='topic_id', doc_col='content_id', score_col='score')
```

## Workflow

1. Score all (query, candidate) pairs with reranker
2. Apply threshold to select positive matches
3. Identify queries with zero positive matches
4. For each empty query, assign the single highest-scoring candidate
5. Concatenate threshold-based matches and fallback matches

## Key Decisions

- **Always use with MAP/recall metrics**: zero predictions = zero recall for that query
- **Top-1 fallback**: simple and effective; top-K fallback may help with F2 metrics
- **Language filter first**: if applying a language-match filter, do it before fallback so fallback respects language constraints
- **Score sorting**: sort descending before groupby().head(1) to guarantee the best candidate

## References

- [LECR-stsb_roberta_base](https://www.kaggle.com/code/yuiwai/lecr-stsb-roberta-base)
- [0.459 | Single Model Inference w/ PostProcessing](https://www.kaggle.com/code/karakasatarik/0-459-single-model-inference-w-postprocessing)
