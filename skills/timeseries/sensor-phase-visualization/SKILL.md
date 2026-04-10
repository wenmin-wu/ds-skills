---
name: timeseries-sensor-phase-visualization
description: EDA visualization of multi-modal sensor data with axvspan shading for labeled behavioral phases and contiguous span detection
domain: timeseries
---

# Sensor Phase Visualization

## Overview

For labeled time-series data with behavioral phases (gesture, transition, activity), visualize multi-modal sensor signals with color-shaded phase regions. Detects contiguous spans from phase labels using index-diff grouping, and overlays them on sensor plots. Essential EDA for understanding sensor-behavior relationships.

## Quick Start

```python
import numpy as np
import matplotlib.pyplot as plt

def shade_phases(ax, timestamps, phase_labels, phase_colors):
    """Overlay colored regions for each behavioral phase.
    
    Args:
        ax: matplotlib axis
        timestamps: array of time values
        phase_labels: array of phase strings per timestep
        phase_colors: dict mapping phase name → color
    """
    for phase, color in phase_colors.items():
        mask = phase_labels == phase
        if not mask.any():
            continue
        indices = np.where(mask)[0]
        # Group contiguous indices into spans
        breaks = np.where(np.diff(indices) != 1)[0]
        spans = np.split(indices, breaks + 1)
        for span in spans:
            t0, t1 = timestamps[span[0]], timestamps[span[-1]]
            ax.axvspan(t0, t1, color=color, alpha=0.25, label=None)

# Usage
fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
phases = {"Gesture": "salmon", "Transition": "lightgray"}
for ax, (name, cols) in zip(axes, sensor_groups.items()):
    ax.plot(df['time'], df[cols])
    shade_phases(ax, df['time'].values, df['phase'].values, phases)
    ax.set_ylabel(name)
```

## Key Decisions

- **Contiguous span detection**: `np.diff(indices) != 1` finds boundaries between non-adjacent labeled regions
- **alpha=0.25**: transparent enough to see signals underneath
- **sharex**: synchronized zoom across modalities for correlation analysis

## References

- Source: [sensor-pulse-viz-eda-for-bfrb-detection](https://www.kaggle.com/code/tarundirector/sensor-pulse-viz-eda-for-bfrb-detection)
- Competition: CMI - Detect Behavior with Sensor Data
