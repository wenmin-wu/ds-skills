---
name: llm-exhaustive-permutation-early-stopping
description: Enumerate all factorial permutations in batches with LLM scoring, tracking the running best and early-stopping when score crosses a known optimality threshold
---

# Exhaustive Permutation with Early Stopping

## Overview

For short sequences (up to 10-12 elements where n! is tractable), enumerate all permutations, score them in batches via a fast LLM scorer, and track the global best. Early-stop when the score drops below a known threshold (e.g., from leaderboard or theoretical bound). Batched evaluation makes this surprisingly fast — 10! = 3.6M permutations at batch_size=256 takes ~14K forward passes.

## Quick Start

```python
import itertools
import math
import numpy as np

def exhaustive_search(words, scorer, batch_size=256, threshold=None):
    best_score = float('inf')
    best_text = None
    total = math.factorial(len(words))
    batch = []
    for i, perm in enumerate(itertools.permutations(words)):
        batch.append(' '.join(perm))
        if len(batch) == batch_size or i == total - 1:
            scores = scorer.score(batch, batch_size=batch_size)
            min_idx = np.argmin(scores)
            if scores[min_idx] < best_score:
                best_score = scores[min_idx]
                best_text = batch[min_idx]
            batch = []
            if threshold and best_score < threshold:
                break
    return best_text, best_score

result, score = exhaustive_search(
    text.split(), scorer, batch_size=256, threshold=475)
```

## Workflow

1. Generate all n! permutations via `itertools.permutations`
2. Accumulate into batches of size B
3. Score each batch with batched LLM perplexity
4. Track running best score and corresponding text
5. Early-stop if score falls below the target threshold

## Key Decisions

- **Feasibility**: practical up to n=10 (3.6M); n=12 (479M) is borderline; n>12 needs heuristics
- **Batch size**: 128-512 balances GPU utilization and memory
- **Threshold**: set from leaderboard probing or known optimal — avoids wasting compute
- **Enumeration order**: reversed order sometimes hits good solutions faster for specific LLMs

## References

- [Brute Force First Sample - Perplexity 470](https://www.kaggle.com/code/cdeotte/brute-force-first-sample-perplexity-470)
