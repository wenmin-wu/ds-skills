---
name: timeseries-scaled-nan-sentinel
description: Encode missing sensor data with a per-modality sentinel value that survives standardization and remains detectable after scaling
domain: timeseries
---

# Scaled NaN Sentinel

## Overview

StandardScaler transforms NaN-replacement values unpredictably. Instead, set a sentinel as a negative multiple of the modality's max value, then precompute what this sentinel becomes after scaling. The model can learn to detect the scaled sentinel as "missing" without NaN handling in the forward pass.

## Quick Start

```python
import numpy as np
from sklearn.preprocessing import StandardScaler

class SentinelScaler:
    def __init__(self, ratio=3.0):
        self.ratio = ratio
        self.scaler = StandardScaler()
        self.sentinel_raw = {}
        self.sentinel_scaled = {}

    def fit(self, data, modality_cols):
        """Fit scaler and compute sentinel values per modality."""
        for name, cols in modality_cols.items():
            max_val = data[cols].max().max()
            self.sentinel_raw[name] = -max_val * self.ratio
        filled = data.copy()
        for name, cols in modality_cols.items():
            filled[cols] = filled[cols].fillna(self.sentinel_raw[name])
        self.scaler.fit(filled)
        # Precompute scaled sentinel
        for name, cols in modality_cols.items():
            dummy = np.full((1, len(filled.columns)), 0.0)
            idx = [filled.columns.get_loc(c) for c in cols]
            for i in idx:
                dummy[0, i] = self.sentinel_raw[name]
            scaled = self.scaler.transform(dummy)
            self.sentinel_scaled[name] = np.mean(scaled[0, idx])

    def transform(self, data, modality_cols):
        filled = data.copy()
        for name, cols in modality_cols.items():
            filled[cols] = filled[cols].fillna(self.sentinel_raw[name])
        return self.scaler.transform(filled)
```

## Key Decisions

- **Negative multiple of max**: ensures sentinel is far from valid data range
- **ratio=3.0**: 3x max value is clearly out of distribution
- **Precompute scaled value**: avoids runtime NaN checks in the model
- **Per-modality**: different sensors have different ranges, need separate sentinels

## References

- Source: [just-changed-the-ensemble-weights](https://www.kaggle.com/code/sasaleaf/just-changed-the-ensemble-weights)
- Competition: CMI - Detect Behavior with Sensor Data
