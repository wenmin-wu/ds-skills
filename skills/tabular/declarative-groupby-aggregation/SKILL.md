---
name: tabular-declarative-groupby-aggregation
description: >
  Config-driven feature factory that generates groupby aggregation features from a declarative spec list, supporting count, mean, var, nunique, cumcount, and custom lambdas.
---
# Declarative GroupBy Aggregation

## Overview

Feature engineering in tabular competitions often requires dozens of groupby aggregations (count by IP, nunique apps per device, variance of channel per OS, etc.). Instead of writing each one manually, define a spec list of `{groupby, select, agg}` dictionaries and loop through them. This pattern is self-documenting, easy to extend, and reduces copy-paste bugs. Used extensively in fraud detection, recommendation, and any multi-entity dataset.

## Quick Start

```python
import pandas as pd
import numpy as np
import gc

GROUPBY_AGGREGATIONS = [
    {'groupby': ['ip'], 'select': 'app', 'agg': 'nunique'},
    {'groupby': ['ip', 'app'], 'select': 'channel', 'agg': 'count'},
    {'groupby': ['ip', 'device', 'os'], 'select': 'app', 'agg': 'cumcount'},
    {'groupby': ['ip'], 'select': 'channel', 'agg': 'var'},
    {'groupby': ['ip', 'app', 'os'], 'select': 'hour', 'agg': 'mean'},
    {'groupby': ['ip'], 'select': 'app', 'agg': lambda x: x.value_counts().head(1).index[0],
     'agg_name': 'mode'},
]

for spec in GROUPBY_AGGREGATIONS:
    agg_name = spec.get('agg_name', spec['agg'].__name__
                        if callable(spec['agg']) else spec['agg'])
    new_feature = '{}_{}_{}'.format('_'.join(spec['groupby']),
                                    agg_name, spec['select'])
    cols = list(set(spec['groupby'] + [spec['select']]))
    gp = df[cols].groupby(spec['groupby'])[spec['select']] \
        .agg(spec['agg']).reset_index() \
        .rename(columns={spec['select']: new_feature})

    if spec['agg'] == 'cumcount':
        df[new_feature] = gp[0].values
    else:
        df = df.merge(gp, on=spec['groupby'], how='left')
    del gp; gc.collect()
```

## Workflow

1. Define a list of spec dicts: `groupby` keys, `select` column, `agg` function
2. Loop through specs, computing each aggregation
3. Merge result back to main DataFrame (or assign directly for cumcount)
4. Delete intermediates and gc.collect() to manage memory

## Key Decisions

- **Naming convention**: Auto-generate feature names from spec for traceability
- **cumcount vs merge**: cumcount produces a per-row sequence number — assign directly, don't merge
- **Memory**: Delete groupby results after merge; process specs in order of memory impact
- **Custom aggs**: Use `agg_name` key for lambdas since `__name__` defaults to `<lambda>`

## References

- [Feature Engineering & Importance Testing](https://www.kaggle.com/code/nanomathias/feature-engineering-importance-testing)
