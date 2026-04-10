---
name: timeseries-event-peak-detection
description: >
  Detects discrete events (state transitions) from continuous predictions using local maxima with minimum-interval constraints.
---
# Event Peak Detection

## Overview

Convert continuous per-timestep predictions into discrete event detections by finding local maxima. Enforce a minimum interval between events to prevent spurious detections. Rank candidates by score and select the top-k based on expected event count (e.g., one onset/wakeup per day). Works with any per-class probability output.

## Quick Start

```python
import numpy as np
from scipy.signal import argrelmax

def detect_events(predictions, min_interval, expected_per_day, n_days):
    """Detect events as local maxima in prediction signal.

    Args:
        predictions: array of shape (T, n_classes), per-timestep probabilities
        min_interval: minimum steps between events (e.g., 12*60*6 for 6 hours)
        expected_per_day: expected events per day per class
        n_days: total days in the series
    """
    events = {}
    for cls in range(predictions.shape[1]):
        # Method 1: scipy argrelmax (fast, clean)
        peaks = argrelmax(predictions[:, cls], order=min_interval)[0]

        # Method 2: manual local max (more control)
        scores = np.zeros(len(predictions))
        for i in range(len(predictions)):
            window = predictions[max(0, i - min_interval):i + min_interval, cls]
            if predictions[i, cls] == window.max():
                scores[i] = predictions[i, cls]

        # Select top candidates by expected count
        n_events = max(1, round(expected_per_day * n_days))
        top_indices = np.argsort(scores)[-n_events:]
        events[cls] = sorted(top_indices)

    return events
```

## Workflow

1. Generate per-timestep probabilities from model (one column per event type)
2. Optionally smooth predictions first (rolling mean or LPF)
3. Find local maxima with `argrelmax(order=min_interval)` or manual scan
4. Rank peaks by score, select top-k based on expected daily count
5. Output event timestamps with confidence scores

## Key Decisions

- **min_interval**: Domain-dependent — 6 hours for sleep events, seconds for heartbeats
- **Expected count**: Use training data statistics (events per day) to set top-k
- **Smoothing first**: Reduces false peaks — apply before peak detection, not after
- **scipy vs manual**: argrelmax is faster; manual allows asymmetric windows

## References

- Child Mind Institute - Detect Sleep States (Kaggle)
- Source: [sleep-critical-point-infer](https://www.kaggle.com/code/werus23/sleep-critical-point-infer)
