---
name: timeseries-multi-gap-lag-diff-features
description: Generate shift/diff features at multiple lag sizes (1,2,3,5,10,20,50,100) over cursor/time/state series, then aggregate statistics per session
---

## Overview

In event-stream data, a single-step diff (`x.diff(1)`) captures local change but misses medium- and long-range dynamics. Generating diffs at a log-spaced ladder of lags (1, 2, 3, 5, 10, 20, 50, 100) gives you a multi-resolution view: step-1 catches individual keystrokes, step-10 catches word-level motion, step-100 catches sentence-level motion. Each lag column is then aggregated with mean/std/min/max per session. One ladder over three series (time, cursor, word count) expands to ~96 features — expensive but consistently useful in keystroke / sensor / clickstream tasks.

## Quick Start

```python
GAPS = [1, 2, 3, 5, 10, 20, 50, 100]

def multi_gap_features(df, id_col='id'):
    for gap in GAPS:
        df[f'up_time_shift{gap}']    = df.groupby(id_col)['up_time'].shift(gap)
        df[f'action_gap{gap}']       = df['down_time'] - df[f'up_time_shift{gap}']

        df[f'cursor_shift{gap}']     = df.groupby(id_col)['cursor_position'].shift(gap)
        df[f'cursor_change{gap}']    = df['cursor_position'] - df[f'cursor_shift{gap}']
        df[f'cursor_abs_change{gap}'] = df[f'cursor_change{gap}'].abs()

        df[f'wc_shift{gap}']  = df.groupby(id_col)['word_count'].shift(gap)
        df[f'wc_change{gap}'] = df['word_count'] - df[f'wc_shift{gap}']

    # Aggregate lag columns per id
    agg_cols = [c for c in df.columns if 'gap' in c or 'change' in c]
    return df.groupby(id_col)[agg_cols].agg(['mean', 'std', 'min', 'max'])
```

## Workflow

1. Sort events by time within each session
2. For each `gap` in the ladder, compute `shift(gap)` within the session, then subtract
3. Take absolute values for columns where direction is noise (cursor jumps)
4. Aggregate each lag-diff column with mean / std / min / max per session
5. Drop redundant columns with correlation > 0.99 — adjacent lags often collapse

## Key Decisions

- **Log-spaced ladder**: doubles coverage vs. linear (1-8) without blowing up feature count.
- **Max lag ≈ √session_length**: for sessions ~10k events, lag 100 is the right ceiling. Going further yields mostly NaN.
- **Aggregate, don't feed raw**: models hate millions of long-format rows. Collapse to session-level features.
- **vs. rolling windows**: rolling is smoother but slower and harder to reason about; multi-gap diffs are discrete and explicit.

## References

- [LGBM (X2) + NN + Fusion](https://www.kaggle.com/code/cody11null/lgbm-x2-nn-fusion)
