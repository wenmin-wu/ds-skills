---
name: tabular-leak-free-loop-features
description: >
  Iterates through rows chronologically to accumulate user statistics, fetching current state before updating to prevent future data leakage.
---
# Leak-Free Loop Features

## Overview

For sequential prediction tasks (knowledge tracing, click prediction), compute running user statistics by looping through rows in time order. The key: fetch the user's current stats BEFORE updating them with the current row's outcome. This ensures each row only sees past data — no future leakage.

## Quick Start

```python
import numpy as np
from collections import defaultdict

def build_loop_features(df, user_col='user_id', target_col='answered_correctly'):
    """Accumulate user stats with strict temporal ordering."""
    sum_dict = defaultdict(float)
    count_dict = defaultdict(int)

    n = len(df)
    user_mean = np.zeros(n, dtype=np.float32)
    user_count = np.zeros(n, dtype=np.int32)

    for i, (uid, target) in enumerate(zip(df[user_col].values, df[target_col].values)):
        # Fetch BEFORE update — this prevents leakage
        if count_dict[uid] > 0:
            user_mean[i] = sum_dict[uid] / count_dict[uid]
        else:
            user_mean[i] = np.nan  # cold start

        user_count[i] = count_dict[uid]

        # Update AFTER fetch
        sum_dict[uid] += target
        count_dict[uid] += 1

    df['user_mean'] = user_mean
    df['user_count'] = user_count
    return df
```

## Workflow

1. Sort data by timestamp (must be chronological)
2. Initialize empty dictionaries for running sums and counts
3. For each row: fetch current user stats → assign as features → update stats with outcome
4. Handle cold start (first interaction) with NaN or global mean
5. Repeat for other entities (content_id, tag, etc.)

## Key Decisions

- **Fetch-then-update**: The ONLY correct order — update-then-fetch leaks the current answer
- **Speed**: Pure Python loops are slow; use numba `@jit` or write in C for >10M rows
- **Cold start**: NaN + fill with global mean, or use a prior (e.g., 0.5 for binary)
- **Multiple entities**: Build separate loop features for user, content, user×content, tag

## References

- Riiid Answer Correctness Prediction (Kaggle)
- Source: [lgbm-with-loop-feature-engineering](https://www.kaggle.com/code/its7171/lgbm-with-loop-feature-engineering)
