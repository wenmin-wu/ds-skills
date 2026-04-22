---
name: llm-stopword-priority-sorting
description: Initialize text ordering by placing stopwords first then content words, producing low-perplexity starting points for combinatorial search over word permutations
---

# Stopword Priority Sorting

## Overview

When minimizing LLM perplexity over word permutations, the initial ordering matters for local search convergence. Sorting stopwords (the, a, is, of) first and content words second produces a surprisingly low starting perplexity because LLMs assign high probability to function words at sequence starts. This simple heuristic gives a 20-40% perplexity reduction over alphabetical sorting as a starting point.

## Quick Start

```python
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words('english'))

def stopword_priority_sort(words):
    stops = sorted([w for w in words if w.lower() in STOP_WORDS])
    content = sorted([w for w in words if w.lower() not in STOP_WORDS])
    return stops + content

initial_order = stopword_priority_sort(text.split())
initial_text = ' '.join(initial_order)
```

## Workflow

1. Split text into words
2. Partition into stopwords and content words
3. Sort each partition alphabetically (or by frequency)
4. Concatenate: stopwords first, then content words
5. Use as initialization for SA, sliding window, or greedy reinsert search

## Key Decisions

- **Why it works**: LLMs model P(w_i | w_{<i}); function words have high conditional probability after other function words
- **Alphabetical within groups**: provides deterministic tie-breaking; frequency-based may be slightly better
- **vs random**: 2-5x lower starting perplexity than random shuffle
- **Limitations**: only useful as initialization — follow with local search for final optimization

## References

- [To Winning - Sort Off](https://www.kaggle.com/code/jazivxt/to-winning-sort-off)
