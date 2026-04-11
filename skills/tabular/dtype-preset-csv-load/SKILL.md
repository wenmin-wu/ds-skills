---
name: tabular-dtype-preset-csv-load
description: >
  Predefines minimal unsigned integer dtypes before CSV loading to cut DataFrame memory usage by 2-4x without any data loss.
---
# Dtype Preset CSV Load

## Overview

Pandas defaults to int64/float64 for all numeric columns, wasting 4–6 bytes per value when the actual range fits in uint8/uint16/uint32. By passing a `dtype` dict to `pd.read_csv()`, you enforce minimal types at load time — before the full-size DataFrame ever exists in memory. On a 200M-row click log, this can mean the difference between fitting in 16GB RAM or not. Combined with `usecols` to skip unneeded columns, this is the first line of defense for large datasets.

## Quick Start

```python
import pandas as pd

# Inspect column ranges first (on a small sample)
sample = pd.read_csv('train.csv', nrows=100_000)
for col in sample.select_dtypes('number').columns:
    print(f"{col}: {sample[col].min()} - {sample[col].max()}")

# Define minimal dtypes based on observed ranges
dtypes = {
    'ip':            'uint32',   # max ~300k → fits uint32
    'app':           'uint16',   # max ~700 → fits uint16
    'device':        'uint16',   # max ~4000
    'os':            'uint16',   # max ~900
    'channel':       'uint16',   # max ~500
    'is_attributed': 'uint8',    # binary 0/1
}

# Load with preset dtypes — 2-4x less memory
train = pd.read_csv(
    'train.csv',
    dtype=dtypes,
    usecols=list(dtypes.keys()) + ['click_time'],
    parse_dates=['click_time'],
)

print(f"Memory: {train.memory_usage(deep=True).sum() / 1e9:.2f} GB")
```

## Workflow

1. Load a small sample (100k rows) to inspect column value ranges
2. Map each column to the smallest dtype that covers its range
3. Pass dtype dict to `pd.read_csv()` along with `usecols`
4. Verify no overflow: `assert df[col].max() < np.iinfo(dtype).max`

## Key Decisions

- **uint vs int**: Use unsigned when values are non-negative — saves 1 bit of range
- **float16**: Risky — only 3 decimal digits of precision. Use float32 as minimum for floats
- **Categorical**: For string columns with few unique values, use `dtype='category'` for 10-50x savings
- **parse_dates**: Datetime columns can't be dtype-optimized — parse separately

## References

- [How to Work with BIG Datasets on 16G RAM (+Dask)](https://www.kaggle.com/code/yuliagm/how-to-work-with-big-datasets-on-16g-ram-dask)
