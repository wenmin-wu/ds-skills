---
name: llm-sliding-window-permutation-search
description: Local search that slides a window of size p across a word sequence, brute-forcing all permutations within each window to minimize an objective like LLM perplexity
---

# Sliding Window Permutation Search

## Overview

For sequences too long for full factorial search, slide a small window (3-5 elements) across the sequence and exhaustively try all permutations within each window position. The prefix and suffix remain fixed while the window contents are reordered. This is O(n * p!) per pass — tractable for p <= 7 — and converges quickly as each window improves its local ordering.

## Quick Start

```python
import itertools

def sliding_window_optimize(sequence, score_fn, window_size=4, skip=1):
    best = sequence[:]
    best_score = score_fn(best)
    improved = True
    while improved:
        improved = False
        for start in range(0, len(best) - window_size + 1):
            end = start + window_size
            prefix = best[:start]
            suffix = best[end:]
            window = best[start:end]
            for perm in itertools.permutations(window):
                candidate = prefix + list(perm) + suffix
                s = score_fn(candidate)
                if s < best_score:
                    best = candidate
                    best_score = s
                    improved = True
    return best, best_score

words = text.split()
result, score = sliding_window_optimize(
    words, lambda w: perplexity(' '.join(w)), window_size=4)
```

## Workflow

1. Start with an initial ordering (e.g., sorted or heuristic-initialized)
2. For each window position `[start:start+p]`, enumerate all p! permutations
3. Score each candidate (prefix + permutation + suffix)
4. Keep the best scoring permutation for that window
5. Repeat passes until no window improves

## Key Decisions

- **Window size**: 3-5 is practical (3!=6, 5!=120, 7!=5040 evaluations per position)
- **Skip parameter**: evaluate every k-th permutation for larger windows to save compute
- **Pass direction**: alternate forward/backward passes to avoid directional bias
- **Convergence**: typically 2-4 passes suffice for near-optimal local ordering

## References

- [To Winning - Sort Off](https://www.kaggle.com/code/jazivxt/to-winning-sort-off)
