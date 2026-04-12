---
name: tabular-chunked-hdf5-streaming
description: >
  Streams large HDF5 files in fixed-size row chunks to compute summary statistics without loading the full dataset into memory.
---
# Chunked HDF5 Streaming

## Overview

Genomics and single-cell datasets often exceed 10GB as HDF5 files — loading them fully into a DataFrame causes OOM. `pd.read_hdf` supports `start`/`stop` parameters for chunk-based reading. Process each chunk independently (compute row sums, nonzero counts, per-group statistics), accumulate results, then discard the chunk. This pattern handles arbitrarily large files with constant memory usage, and is essential for EDA and feature engineering on competition datasets stored in HDF5 format.

## Quick Start

```python
import pandas as pd
import numpy as np

def stream_hdf5_stats(filepath, chunksize=5000):
    """Compute per-row summary stats from large HDF5 in chunks."""
    summaries = []
    start = 0
    while True:
        chunk = pd.read_hdf(filepath, start=start, stop=start + chunksize)
        if len(chunk) == 0:
            break
        summaries.append(pd.DataFrame({
            'row_sum': chunk.sum(axis=1),
            'nonzero_count': (chunk != 0).sum(axis=1),
            'row_mean': chunk.mean(axis=1),
        }, index=chunk.index))
        if len(chunk) < chunksize:
            break
        start += chunksize
    return pd.concat(summaries)

stats = stream_hdf5_stats('train_multi_inputs.h5', chunksize=5000)
```

## Workflow

1. Open HDF5 file with `pd.read_hdf(path, start=, stop=)`
2. Process each chunk: compute row-level or column-level statistics
3. Accumulate results in a list, then `pd.concat` at the end
4. Break when chunk is smaller than chunksize (end of file)

## Key Decisions

- **Chunksize**: 5000-10000 rows balances memory vs overhead; tune to available RAM
- **Column-wise stats**: For column means/stds, use running accumulators (Welford's algorithm)
- **Sparse conversion**: Convert each chunk to `scipy.sparse.csr_matrix` if >90% zeros
- **vs Dask**: Chunks are simpler and more predictable; Dask adds overhead for single-file reads

## References

- [MSCI EDA which makes sense](https://www.kaggle.com/code/ambrosm/msci-eda-which-makes-sense)
