---
name: llm-greedy-word-reinsert-search
description: Greedy local search that removes one element from a fixed position and re-inserts it at every possible index, keeping the best improvement per round
---

# Greedy Word Reinsert Search

## Overview

A lightweight local search for sequence ordering: pick a position, remove the element, then try inserting it at every other position. Keep the best insertion if it improves the score. Sweep through all positions in one round. This is O(n^2) evaluations per round and converges in 3-10 rounds. Effective as a polishing step after coarser heuristics.

## Quick Start

```python
def greedy_reinsert(sequence, score_fn, max_rounds=10):
    best = sequence[:]
    best_score = score_fn(best)
    for _ in range(max_rounds):
        improved = False
        for pos in range(len(best)):
            elem = best[pos]
            remaining = best[:pos] + best[pos+1:]
            for insert_at in range(len(remaining) + 1):
                candidate = remaining[:insert_at] + [elem] + remaining[insert_at:]
                s = score_fn(candidate)
                if s < best_score:
                    best = candidate
                    best_score = s
                    improved = True
                    break
        if not improved:
            break
    return best, best_score

words = text.split()
result, score = greedy_reinsert(
    words, lambda w: perplexity(' '.join(w)))
```

## Workflow

1. For each position in the sequence, extract the element
2. Try inserting it at every other position (0 to n-1)
3. Score each candidate sequence
4. Accept the first improvement (greedy) or the best improvement (steepest descent)
5. Repeat rounds until no improvement found

## Key Decisions

- **First-improvement vs best-improvement**: first is faster per round; best finds larger gains
- **Sweep direction**: alternate forward/backward to avoid ordering bias
- **Position selection**: prioritize boundary positions (first/last) for bigger perplexity gains
- **Combining with SA**: use reinsert as a post-processing step after simulated annealing

## References

- [Diminutive Effort](https://www.kaggle.com/code/jazivxt/diminutive-effort)
