---
name: tabular-yearly-partitioned-groupby
description: Split a multi-year table into per-year partitions, run the same groupby aggregation on each, then concat and gc — a pure-pandas map-reduce that survives 100M+ rows on a 16GB kernel
---

## Overview

A naive `df.groupby(['year','month','day','store'])['sales'].agg(['sum','count'])` on a 125M-row Favorita train set hangs or OOMs a 16GB kernel because the intermediate hash table dwarfs available memory. Slicing by year first keeps each partition's hash table small, runs aggregation per partition, and `pd.concat`s the results. This is map-reduce implemented in pure pandas — no Dask, no HDF5 streaming — and it's usually the cheapest fix when a single `groupby` kills your kernel.

## Quick Start

```python
import pandas as pd
import gc

def aggregate_partition(df):
    day_store = df.groupby(['Year','Month','Day','store_nbr'],
                           as_index=False)['unit_sales'].agg(['sum','count'])
    day_item  = df.groupby(['Year','Month','Day','item_nbr'],
                           as_index=False)['unit_sales'].agg(['sum','count'])
    return day_store, day_item

store_parts, item_parts = [], []
for y in sorted(train['Year'].unique()):
    ds, di = aggregate_partition(train.loc[train['Year'] == y])
    store_parts.append(ds); item_parts.append(di)
    gc.collect()

day_store = pd.concat(store_parts); day_item = pd.concat(item_parts)
del store_parts, item_parts; gc.collect()
```

## Workflow

1. Ensure the partition key (e.g. `Year`) is a small int so slicing is fast
2. Write a pure function that takes a DataFrame and returns all desired rollups for one partition
3. Loop over unique partition keys; slice with `.loc`, aggregate, append results
4. `pd.concat` the per-partition results along axis 0
5. `del` intermediate lists and call `gc.collect()` explicitly — pandas does not release memory eagerly
6. Checkpoint the aggregated frames to parquet so downstream EDA never re-touches the raw table

## Key Decisions

- **Partition on the coarsest key that still fits one slice in memory**: year for Favorita, month for denser data, store-hash for really huge.
- **Run ALL rollups inside one partition pass**: loading the slice is expensive; compute every aggregate you'll need while it's hot.
- **Explicit `gc.collect()`**: without it, the freed DataFrames linger and the next iteration adds to peak memory.
- **Not a replacement for Dask**: if a single partition doesn't fit, you need real out-of-core. This pattern solves the "one groupby is too big but one slice fits" case, which is most of the time.

## References

- [Memory optimization and EDA on entire dataset](https://www.kaggle.com/code/tunguz/memory-optimization-and-eda-on-entire-dataset)
