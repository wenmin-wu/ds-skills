---
name: tabular-next-click-time-delta
description: >
  Computes seconds until the next event within a group using diff().shift(-1) on sorted timestamps, capturing user behavior velocity.
---
# Next-Click Time Delta

## Overview

In click-stream, transaction, and event-log data, the time gap between consecutive events from the same entity (user, IP, device) is a powerful behavioral signal. Short gaps indicate bots or burst activity; long gaps indicate re-engagement. This technique computes forward-looking time deltas via `groupby().transform(lambda x: x.diff().shift(-1))` — no join needed, stays aligned with the original index. Can be computed for multiple group granularities (IP only, IP+app, IP+app+device+os).

## Quick Start

```python
import pandas as pd

NEXT_CLICK_GROUPS = [
    {'groupby': ['ip']},
    {'groupby': ['ip', 'app']},
    {'groupby': ['ip', 'app', 'device', 'os', 'channel']},
]

for spec in NEXT_CLICK_GROUPS:
    feature_name = '{}_nextClick'.format('_'.join(spec['groupby']))
    df[feature_name] = (
        df.groupby(spec['groupby'])['click_time']
        .transform(lambda x: x.diff().shift(-1))
        .dt.seconds
    )

# Previous click (backward-looking)
for spec in NEXT_CLICK_GROUPS:
    feature_name = '{}_prevClick'.format('_'.join(spec['groupby']))
    df[feature_name] = (
        df.groupby(spec['groupby'])['click_time']
        .transform(lambda x: x.diff())
        .dt.seconds
    )
```

## Workflow

1. Sort DataFrame by timestamp (usually already sorted in event logs)
2. Define group granularities from broad (IP) to narrow (IP+app+device+os+channel)
3. Compute forward delta: `diff().shift(-1)` gives time to next event
4. Compute backward delta: `diff()` gives time since previous event
5. Extract `.dt.seconds` for numeric feature; NaN for first/last in group

## Key Decisions

- **Forward vs backward**: Both are useful — forward predicts intent, backward measures recency
- **Granularity**: Broader groups (IP only) capture overall velocity; narrow groups capture specific behavior
- **NaN handling**: First/last rows per group are NaN — fill with -1 or large value, or let tree models handle
- **Memory**: `.transform()` avoids merge; for very large data, compute per-group in chunks

## References

- [Feature Engineering & Importance Testing](https://www.kaggle.com/code/nanomathias/feature-engineering-importance-testing)
