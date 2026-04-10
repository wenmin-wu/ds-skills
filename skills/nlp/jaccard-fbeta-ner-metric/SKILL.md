---
name: nlp-jaccard-fbeta-ner-metric
description: >
  Computes micro F-beta for NER where true positives are determined by Jaccard word-overlap >= threshold rather than exact string match.
---
# Jaccard F-Beta NER Metric

## Overview

Exact string match is too strict for NER evaluation — "National Health Interview Survey" vs "National Health Interview Survey (NHIS)" should count as a match. Jaccard F-beta uses word-level Jaccard similarity with a threshold (e.g., 0.5) to determine true positives, then computes micro F-beta. Beta < 1 (e.g., 0.5) penalizes false positives more than false negatives.

## Quick Start

```python
import numpy as np

def jaccard_similarity(s1, s2):
    w1, w2 = set(s1.lower().split()), set(s2.lower().split())
    return len(w1 & w2) / len(w1 | w2) if w1 | w2 else 0.0

def jaccard_fbeta(y_true, y_pred, beta=0.5, jaccard_thresh=0.5):
    tp = fp = fn = 0
    for gt_list, pred_list in zip(y_true, y_pred):
        pred_remaining = list(pred_list)
        for gt in gt_list:
            scores = [jaccard_similarity(gt, p) for p in pred_remaining]
            if scores:
                best_idx = int(np.argmax(scores))
                if scores[best_idx] >= jaccard_thresh:
                    pred_remaining.pop(best_idx)
                    tp += 1
                else:
                    fn += 1
            else:
                fn += 1
        fp += len(pred_remaining)
    b2 = beta ** 2
    return tp * (1 + b2) / (tp * (1 + b2) + fp + fn * b2) if tp > 0 else 0.0
```

## Workflow

1. For each document, pair ground-truth entities with predicted entities
2. Greedily match: find the prediction with highest Jaccard vs each ground truth
3. If Jaccard >= threshold, count as TP and remove from candidates
4. Remaining unmatched predictions = FP; unmatched ground truths = FN
5. Compute micro F-beta across all documents

## Key Decisions

- **Jaccard threshold**: 0.5 is standard for partial match; 1.0 = exact match
- **Beta**: 0.5 penalizes FP more (precision-oriented); 2.0 penalizes FN more (recall-oriented)
- **Greedy vs optimal matching**: Greedy is standard for this metric; optimal (Hungarian) is expensive
- **Normalization**: Lowercase and strip punctuation before comparison

## References

- [External_Datasets_Matching + MLM](https://www.kaggle.com/code/chienhsianghung/external-datasets-matching-mlm)
