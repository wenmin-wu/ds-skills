---
name: cv-deterministic-hash-partitioning
description: Partition a large dataset into N balanced shards using integer key modulo arithmetic for reproducible, class-interleaved splits across CSV files
---

# Deterministic Hash Partitioning

## Overview

When a dataset is too large to fit in memory (e.g., 50M doodles across 340 classes), deterministic hash partitioning splits it into N shards using `key_id % N`. Each shard gets a balanced mix of all classes, is reproducible without storing the split, and can be processed independently. Combined with streaming append, this builds sharded files without loading the full dataset.

## Quick Start

```python
import pandas as pd
import numpy as np
from tqdm import tqdm

N_SHARDS = 100
categories = [...]  # list of 340 class names

for class_idx, category in enumerate(tqdm(categories)):
    df = pd.read_csv(f"train_{category}.csv", nrows=30000)
    df["label"] = class_idx
    df["shard"] = (df["key_id"] // 10**7) % N_SHARDS

    for k in range(N_SHARDS):
        chunk = df[df["shard"] == k].drop(["key_id", "shard"], axis=1)
        mode = "w" if class_idx == 0 else "a"
        header = class_idx == 0
        chunk.to_csv(f"train_shard_{k}.csv.gz",
                     mode=mode, header=header, index=False,
                     compression="gzip")
```

## Workflow

1. For each class file, read a fixed number of rows (balanced sampling)
2. Compute shard assignment: `key_id // 10^7 % N_SHARDS`
3. Append each shard's rows to the corresponding output file
4. After all classes are processed, each shard contains a balanced mix
5. Shuffle within each shard (add random column, sort, drop)
6. Use shards as independent training chunks for generators

## Key Decisions

- **Hash function**: integer modulo is fast and deterministic; use `// 10^7` to avoid sequential correlation
- **N shards**: 100 gives ~300 rows per class per shard with 30K samples/class — small enough for chunked reading
- **Compression**: gzip each shard to reduce disk I/O (3-5x compression on CSV)
- **Shuffle within shard**: essential — append order groups by class; random sort interleaves them
- **Reproducibility**: same key_id always maps to same shard, no random seed dependency

## References

- [Shuffle CSVs](https://www.kaggle.com/code/gaborfodor/shuffle-csvs)
