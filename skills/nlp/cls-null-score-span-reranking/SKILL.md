---
name: nlp-cls-null-score-span-reranking
description: >
  Reranks candidate answer spans by subtracting the CLS token's start+end logit sum as a null-answer baseline score.
---
# CLS Null-Score Span Reranking

## Overview

In extractive QA, the CLS token's span logits represent the model's confidence that no answer exists. By subtracting the CLS score (start_logits[0] + end_logits[0]) from each candidate span's score, you get a relative measure of how much better the span is than "no answer." This naturally handles unanswerable questions — if no span exceeds the CLS baseline, the question is unanswerable.

## Quick Start

```python
import numpy as np

def rerank_spans(start_logits, end_logits, start_indices, end_indices):
    """Score spans relative to CLS null-answer baseline.

    Args:
        start_logits: (seq_len,) start position logits
        end_logits: (seq_len,) end position logits
        start_indices: top-k start position candidates
        end_indices: top-k end position candidates
    Returns:
        list of (score, start, end) sorted descending
    """
    cls_score = start_logits[0] + end_logits[0]
    candidates = []
    for s in start_indices:
        for e in end_indices:
            if e >= s and (e - s) < 30:  # max span length
                span_score = start_logits[s] + end_logits[e]
                score = span_score - cls_score
                candidates.append((score, s, e))
    return sorted(candidates, reverse=True)

# Positive score = span is better than null; negative = unanswerable
ranked = rerank_spans(start_logits, end_logits, top_starts, top_ends)
if ranked and ranked[0][0] > 0:
    best_start, best_end = ranked[0][1], ranked[0][2]
```

## Workflow

1. Extract start/end logits from the QA model
2. Compute CLS null score: `start_logits[0] + end_logits[0]`
3. For each candidate span, compute: `span_score - cls_score`
4. Sort candidates by relative score descending
5. If best score <= 0, predict "no answer"

## Key Decisions

- **Threshold**: Score > 0 is the natural cutoff; tune on validation for optimal F1
- **Max span length**: Cap at 20-50 tokens to avoid degenerate long spans
- **Top-k**: Take top 20 start and end indices to limit candidate count
- **vs learned threshold**: A simple score > 0 works well; learned thresholds add marginal gains

## References

- [BERT Joint Baseline Notebook](https://www.kaggle.com/code/prokaj/bert-joint-baseline-notebook)
- [BERT Joint](https://www.kaggle.com/code/mmmarchetti/bert-joint)
