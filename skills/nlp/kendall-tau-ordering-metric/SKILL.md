---
name: nlp-kendall-tau-ordering-metric
description: >
  Evaluates predicted sequence ordering quality using Kendall Tau correlation via efficient O(n log n) inversion counting.
---
# Kendall Tau Ordering Metric

## Overview

For tasks that predict the order of items (cell ordering, passage ranking, document sorting), you need a metric that measures how close the predicted permutation is to the ground truth. Kendall Tau counts the number of pairwise inversions (swapped pairs) and normalizes to [-1, 1]. A score of 1 means perfect order, 0 is random, -1 is fully reversed. Uses bisect-based inversion counting for O(n log n) per sample.

## Quick Start

```python
from bisect import bisect

def count_inversions(a):
    inversions = 0
    sorted_so_far = []
    for i, u in enumerate(a):
        j = bisect(sorted_so_far, u)
        inversions += i - j
        sorted_so_far.insert(j, u)
    return inversions

def kendall_tau(ground_truth, predictions):
    total_inversions = 0
    total_2max = 0
    for gt, pred in zip(ground_truth, predictions):
        ranks = [gt.index(x) for x in pred]
        total_inversions += count_inversions(ranks)
        n = len(gt)
        total_2max += n * (n - 1)
    return 1 - 4 * total_inversions / total_2max
```

## Workflow

1. For each sample, map predicted order to ground-truth rank indices
2. Count pairwise inversions in the rank sequence using bisect insertion
3. Normalize: `1 - 4 * inversions / (n * (n - 1))`
4. Average across all samples

## Key Decisions

- **vs Spearman**: Kendall Tau is more robust to outliers; Spearman is smoother
- **Complexity**: O(n log n) via bisect, not O(n^2) naive counting
- **Ties**: This implementation assumes no ties; use scipy for tie-corrected version
- **Alternative**: `scipy.stats.kendalltau` for single pairs, but this batches efficiently

## References

- [Getting Started with AI4Code](https://www.kaggle.com/code/ryanholbrook/getting-started-with-ai4code)
- [AI4Code Pytorch DistilBert Baseline](https://www.kaggle.com/code/aerdem4/ai4code-pytorch-distilbert-baseline)
