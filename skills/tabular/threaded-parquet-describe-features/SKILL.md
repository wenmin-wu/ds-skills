---
name: tabular-threaded-parquet-describe-features
description: Parallel-load per-subject parquet time-series files with ThreadPoolExecutor and flatten describe() statistics into tabular feature vectors
---

# Threaded Parquet Describe Features

## Overview

When time-series data is stored as one parquet file per subject (common in wearable/sensor competitions), sequential loading is slow. Use `ThreadPoolExecutor` to read all files in parallel, compute `df.describe()` per subject, and flatten the 8-stat-per-column summary into a single feature row. Converts variable-length time-series into fixed-width tabular features in seconds.

## Quick Start

```python
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

def process_subject(subject_dir, base_dir):
    path = os.path.join(base_dir, subject_dir, 'part-0.parquet')
    df = pd.read_parquet(path)
    df = df.drop(columns=['step'], errors='ignore')
    stats = df.describe().values.reshape(-1)
    sid = subject_dir.split('=')[-1]
    return stats, sid

def load_timeseries_features(base_dir):
    subjects = os.listdir(base_dir)
    with ThreadPoolExecutor() as pool:
        results = list(pool.map(
            lambda s: process_subject(s, base_dir), subjects))
    stats, ids = zip(*results)
    n_feats = len(stats[0])
    df = pd.DataFrame(list(stats),
                       columns=[f'ts_stat_{i}' for i in range(n_feats)])
    df['id'] = ids
    return df

ts_features = load_timeseries_features('data/series_train.parquet/')
train = train.merge(ts_features, on='id', how='left')
```

## Workflow

1. List all subject directories under the parquet base path
2. `ThreadPoolExecutor.map()` reads each parquet and computes `describe()`
3. Flatten the 8-row x N-column describe output into a 1D vector per subject
4. Assemble into a DataFrame and merge with main table on subject ID

## Key Decisions

- **ThreadPoolExecutor over ProcessPool**: parquet reads are I/O-bound, thread pool avoids pickling overhead
- **describe() gives 8 stats**: count, mean, std, min, 25%, 50%, 75%, max — good baseline coverage
- **Drop `step` column**: monotonic index adds no information
- **Naming**: generic `ts_stat_i` columns; map back via `(stat_idx, col_name)` if interpretability needed

## References

- [CMI | Best Single Model](https://www.kaggle.com/code/abdmental01/cmi-best-single-model)
