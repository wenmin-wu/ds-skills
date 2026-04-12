---
name: timeseries-burst-rle-detection
description: Detect P-bursts (fast-typing runs) and R-bursts (consecutive revisions) via polars run-length encoding over boolean event conditions
---

## Overview

A "burst" is a contiguous run of events that share a property — consecutive keystrokes with <2s gaps (P-burst / production burst), or consecutive Remove/Cut events (R-burst / revision burst). Bursts capture writing *flow* rather than aggregate rate. They are expensive to compute naively but trivial with `rle_id()` in polars: number each run, then aggregate run lengths per session. Used as features, burst statistics (mean/std/max length, count) add 10-20 highly-informative features in keystroke-based writing-quality tasks.

## Quick Start

```python
import polars as pl

# P-bursts: runs where time_diff < 2s
def p_bursts(df):
    df = df.with_columns((pl.col('time_diff') < 2).alias('fast'))
    df = df.with_columns(
        pl.when(pl.col('fast') & pl.col('fast').is_last())
          .then(pl.count())
          .over(pl.col('fast').rle_id())
          .alias('P_burst_len')
    )
    return df.group_by('id').agg(
        pl.col('P_burst_len').mean().alias('pburst_mean'),
        pl.col('P_burst_len').std().alias('pburst_std'),
        pl.col('P_burst_len').max().alias('pburst_max'),
        pl.col('P_burst_len').count().alias('pburst_count'),
    )

# R-bursts: runs of activity == 'Remove/Cut'
def r_bursts(df):
    df = df.with_columns((pl.col('activity') == 'Remove/Cut').alias('rev'))
    df = df.with_columns(
        pl.when(pl.col('rev') & pl.col('rev').is_last())
          .then(pl.count())
          .over(pl.col('rev').rle_id())
          .alias('R_burst_len')
    )
    return df.group_by('id').agg(pl.col('R_burst_len').max().alias('rburst_max'))
```

## Workflow

1. Add a boolean column marking the burst condition (fast typing, revision, pause, etc.)
2. Compute `rle_id()` over that column — each contiguous True/False run gets a unique id
3. Count events per run-id with `pl.count().over(rle_id)` — this gives the run length for every event
4. Aggregate the length column per session (mean, std, max, count of runs)
5. Join features back onto the session-level feature matrix

## Key Decisions

- **polars over pandas**: `rle_id` is built-in and fast; in pandas you'd need `(col != col.shift()).cumsum()`, which is slower and less readable.
- **Multiple burst types**: don't stop at P/R — also try long-pause bursts, same-key bursts, same-paragraph bursts. Each adds orthogonal signal.
- **Aggregate the lengths, not just counts**: `max_burst_len` is often the single most predictive feature because it captures peak flow.
- **`is_last()` filter**: only record the length on the terminal row of each run to avoid double-counting.

## References

- [LGBM (X2) + NN](https://www.kaggle.com/code/cody11null/lgbm-x2-nn)
