---
name: tabular-transitive-match-closure
description: >
  Post-processes entity match predictions to enforce symmetry (A→B implies B→A) and transitivity (A→B, B→C implies A→C) via graph closure.
---
# Transitive Match Closure

## Overview

Binary entity matching classifiers predict pairs independently, producing inconsistent match sets: A may match B without B matching A, or A matches B and B matches C but A doesn't match C. Transitive match closure fixes this by first enforcing symmetry (bidirectional links), then propagating matches through connected components. This consistently improves recall in entity deduplication and record linkage tasks.

## Quick Start

```python
def symmetric_closure(id2matches):
    """If A matches B, ensure B matches A."""
    for base, matches in list(id2matches.items()):
        for m in matches:
            if base not in id2matches.get(m, []):
                id2matches.setdefault(m, [base]).append(base)
    return id2matches

def transitive_closure(id2matches):
    """If A matches B and B matches C, add C to A's matches."""
    changed = True
    while changed:
        changed = False
        for base, matches in list(id2matches.items()):
            expanded = set(matches)
            for m in matches:
                expanded.update(id2matches.get(m, []))
            expanded.discard(base)
            if len(expanded) > len(set(matches)):
                id2matches[base] = list(expanded)
                changed = True
    return id2matches

# Apply to predictions
id2matches = dict(zip(df["id"], df["matches"].str.split()))
id2matches = symmetric_closure(id2matches)
id2matches = transitive_closure(id2matches)
df["matches"] = df["id"].map(lambda x: " ".join(id2matches[x]))
```

## Workflow

1. Parse raw predictions into an ID → matches dictionary
2. Enforce symmetry: for each A→B link, add B→A
3. Enforce transitivity: propagate matches through connected components
4. Rebuild prediction strings from the closed match sets

## Key Decisions

- **Symmetry only vs full closure**: Symmetry is safe; transitivity can over-merge if classifier has false positives
- **Iteration limit**: Cap transitive passes to prevent runaway merging on noisy predictions
- **Confidence filtering**: Only propagate high-confidence matches to limit error cascading
- **Union-Find alternative**: For large datasets, use disjoint-set (union-find) for O(n) closure

## References

- [[Foursquare] LGB + Catboost](https://www.kaggle.com/code/felipefonte99/foursquare-lgb-catboost)
- [Foursquare - LightGBM Baseline](https://www.kaggle.com/code/ryotayoshinobu/foursquare-lightgbm-baseline)
