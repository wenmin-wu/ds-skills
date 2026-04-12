---
name: cv-combinatorial-deletion-point-matching
description: Match two unequal-length point sets by enumerating which points to drop from the longer set, searching over deletion combinations
---

## Overview

When detector output and ground-truth point sets differ in length (missing detections, extra false positives), standard Hungarian matching forces a 1-to-1 assignment and silently mis-pairs points. A cleaner approach: if the length difference is small, enumerate all possible ways to delete `k = |A| - |B|` points from the longer set, compute the distance for each subset, and pick the minimum. For `C(N, k)` combinations that blow up, sample a fixed max-iter budget (e.g. 2000). This handles false positives and missing detections in a single pass without tuning Hungarian cost matrices.

## Quick Start

```python
import itertools, random
import numpy as np

def norm_arr(a):
    a = a - a.min()
    return a / (a.max() + 1e-9)

def match_unequal(a1, a2, max_iter=2000):
    """a1 is the longer set. Returns (distance, deletion_indices)."""
    len_diff = len(a1) - len(a2)
    if len_diff < 0:
        raise ValueError("a1 must be >= len(a2)")
    a2n = norm_arr(np.asarray(a2, dtype=float))
    if len_diff == 0:
        return np.linalg.norm(norm_arr(np.asarray(a1, dtype=float)) - a2n), ()

    del_list = list(itertools.combinations(range(len(a1)), len_diff))
    if len(del_list) > max_iter:
        del_list = random.sample(del_list, max_iter)

    best = (float('inf'), None)
    for idx in del_list:
        kept = norm_arr(np.delete(a1, list(idx)).astype(float))
        d = np.linalg.norm(kept - a2n)
        if d < best[0]:
            best = (d, idx)
    return best
```

## Workflow

1. Confirm which set is longer — the algorithm only deletes from the longer one
2. Compute `len_diff = |longer| - |shorter|` — this is the number of deletions per candidate
3. Generate `C(N, len_diff)` deletion subsets; cap at `max_iter` via random sampling
4. For each subset, delete those indices, normalize, compute distance to the shorter set
5. Return the minimum-distance subset — the deleted indices are the "false positives"

## Key Decisions

- **Random-sample fallback**: when `C(N, k) > max_iter`, uniform sampling preserves the minimum well enough in practice for k ≤ 3.
- **Works on 1D sorted arrays**: combine with rotation-search 1D projections for cheap alignment.
- **vs. Hungarian matching**: Hungarian forces 1-to-1 pairing; this method supports uneven sets natively.
- **Limit to small `len_diff`**: if the detector drops >5 points, pre-filter by confidence before matching.

## References

- [Tuning DeepSort + Helmet Mapping](https://www.kaggle.com/code/its7171/tuning-deepsort-helmet-mapping)
- [Helper Code + Helmet Mapping + Deepsort](https://www.kaggle.com/code/robikscube/helper-code-helmet-mapping-deepsort)
