---
name: tabular-confidence-weighted-rate-encoding
description: >
  Encodes categorical groups by their target rate scaled by a log-confidence factor, smoothing unreliable rates from low-frequency groups toward zero.
---
# Confidence-Weighted Rate Encoding

## Overview

Target/mean encoding maps each categorical value to its average target rate, but low-frequency categories produce noisy estimates. Confidence-weighted rate encoding multiplies the raw rate by `min(1, log(count) / log(threshold))`, smoothing rare groups toward zero while preserving high-frequency estimates. This is a lightweight alternative to Bayesian target encoding that avoids specifying a prior distribution — the log-confidence factor acts as an implicit regularizer.

## Quick Start

```python
import numpy as np
import pandas as pd

LOG_GROUP = np.log(100000)  # confidence saturates at 100k observations

def confidence_weighted_rate(x):
    """Rate × min(1, log(count)/log(threshold))."""
    rate = x.sum() / float(x.count())
    confidence = min(1.0, np.log(x.count()) / LOG_GROUP)
    return rate * confidence

# Single group
df['ip_conf_rate'] = df.groupby('ip')['target'] \
    .transform(confidence_weighted_rate)

# Multi-key groups
GROUP_SPECS = [
    ['ip'],
    ['ip', 'app'],
    ['ip', 'device', 'os'],
    ['app', 'channel'],
]
for cols in GROUP_SPECS:
    feat_name = '_'.join(cols) + '_confRate'
    df[feat_name] = df.groupby(cols)['target'] \
        .transform(confidence_weighted_rate)
```

## Workflow

1. Choose a confidence threshold (e.g., 100,000) — groups with this many samples get full weight
2. For each group, compute raw target rate
3. Multiply by `min(1, log(count) / log(threshold))`
4. Apply to multiple group granularities for richer signal

## Key Decisions

- **Threshold**: Higher = more conservative smoothing. 100k for large datasets; 1000 for small
- **Leak prevention**: Compute on train folds only, merge onto val/test — same as target encoding
- **vs Bayesian TE**: Simpler, no prior needed; Bayesian TE is theoretically better but heavier
- **Log scale**: Log dampens the effect of very large groups — a group with 1M rows gets similar confidence to 100k

## References

- [Feature Engineering & Importance Testing](https://www.kaggle.com/code/nanomathias/feature-engineering-importance-testing)
