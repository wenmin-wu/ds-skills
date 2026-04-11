---
name: nlp-simulated-annealing-query-optimization
description: >
  Uses simulated annealing to select the optimal subset of candidate terms for a boolean search query, maximizing a retrieval metric like AP@K.
---
# Simulated Annealing Query Optimization

## Overview

Boolean search queries (used in patent search, legal discovery, academic retrieval) require selecting which terms to include from a candidate set. The search space is combinatorial — 2^N subsets for N candidate terms. Simulated annealing efficiently explores this space by randomly flipping term inclusion bits and accepting worse solutions with decreasing probability. The energy function evaluates query quality against a full-text index (Whoosh, Elasticsearch), typically using AP@K or recall@K.

## Quick Start

```python
import numpy as np
from simanneal import Annealer

class QueryOptimizer(Annealer):
    def __init__(self, candidates, searcher, targets, **kwargs):
        self.candidates = candidates       # list of search terms
        self.searcher = searcher           # full-text index searcher
        self.targets = targets             # ground-truth document IDs
        initial_state = np.random.binomial(1, 0.5, len(candidates))
        super().__init__(initial_state, **kwargs)

    def move(self):
        """Flip one random term's inclusion bit."""
        idx = np.random.randint(len(self.state))
        self.state[idx] = 1 - self.state[idx]

    def energy(self):
        """Negative AP@50 — lower is better for annealer."""
        active = [t for t, use in zip(self.candidates, self.state) if use]
        if not active:
            return 0.0  # empty query = worst score
        query = " OR ".join(active)
        results = self.searcher.search(query, limit=50)
        return -average_precision(results, self.targets, k=50)

optimizer = QueryOptimizer(candidates, searcher, targets)
optimizer.steps = 2000
optimizer.Tmax = 1.0
optimizer.Tmin = 0.001
best_state, best_energy = optimizer.anneal()
best_terms = [t for t, u in zip(candidates, best_state) if u]
```

## Workflow

1. Generate candidate terms via TF-IDF top-k, LLM extraction, or vocabulary analysis
2. Build a full-text search index over the document corpus
3. Define energy function: negative retrieval metric (AP@K, recall@K)
4. Run simulated annealing — each move flips one term on/off
5. Extract the optimal term subset from the best state

## Key Decisions

- **Candidate set size**: 15–30 terms balances search space vs coverage
- **Temperature schedule**: Start at 1.0, decay to 0.001 over 1000–5000 steps
- **Move operator**: Single bit flip is simplest; swap (flip one on + one off) is also effective
- **Time budget**: 5–15 seconds per query is typical for competition settings

## References

- [USPTO - Annealing [LB = 0.31]](https://www.kaggle.com/code/andrey67/uspto-annealing-lb-0-31)
- [USPTO-Simulated-Annealing-Baseline](https://www.kaggle.com/code/tubotubo/uspto-simulated-annealing-baseline)
