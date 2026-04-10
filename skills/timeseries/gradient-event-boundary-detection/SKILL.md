---
name: timeseries-gradient-event-boundary-detection
description: Detect event start/end boundaries in time series by finding extrema of the first derivative (steepest gradient points)
domain: timeseries
---

# Gradient Event Boundary Detection

## Overview

For events with gradual onset and offset (transits, dips, ramps), locate boundaries by computing the first derivative and finding its extrema. The steepest descent marks event start (ingress); steepest ascent marks event end (egress). Works without threshold tuning.

## Quick Start

```python
import numpy as np

def detect_event_boundaries(signal, search_start=None, search_end=None):
    """Find event ingress/egress via gradient extrema.
    
    Args:
        signal: 1D array of the time series
        search_start/end: optional tuple (lo, hi) to constrain search region
    Returns:
        (ingress_idx, egress_idx)
    """
    event_min = np.argmin(signal)
    # Search for steepest descent before minimum
    pre = signal[:event_min]
    grad_pre = np.gradient(pre)
    ingress = np.argmin(grad_pre)  # most negative gradient
    # Search for steepest ascent after minimum
    post = signal[event_min:]
    grad_post = np.gradient(post)
    egress = np.argmax(grad_post) + event_min
    return ingress, egress
```

## Key Decisions

- **Split at minimum**: separates ingress search (before) from egress search (after)
- **No threshold needed**: uses argmin/argmax of gradient — self-calibrating
- **Smooth first**: apply Savitzky-Golay or rolling mean before gradient if signal is noisy
- **Generalizable**: works for any dip/peak event — not just astronomical transits

## References

- Source: [ariel-only-correlation](https://www.kaggle.com/code/sergeifironov/ariel-only-correlation)
- Competition: NeurIPS - Ariel Data Challenge 2024
