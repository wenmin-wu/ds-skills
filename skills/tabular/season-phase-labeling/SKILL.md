---
name: tabular-season-phase-labeling
description: Map calendar dates to categorical season phases (offseason, preseason, regular, postseason) using np.select with boundary date conditions
---

# Season Phase Labeling

## Overview

Sports, retail, and event-driven datasets exhibit distinct behavioral regimes tied to calendar phases. Mapping each date to its season phase (offseason, preseason, regular season, postseason) creates a powerful categorical feature that captures regime-dependent patterns. Using `np.select` with ordered conditions and boundary dates handles complex multi-phase calendars cleanly.

## Quick Start

```python
import numpy as np

conditions = [
    df["date"] < df["preseason_start"],
    df["date"] < df["regular_start"],
    df["date"] <= df["regular_end"],
    df["date"] < df["postseason_start"],
    df["date"] <= df["postseason_end"],
    df["date"] > df["postseason_end"],
]
labels = ["offseason", "preseason", "regular", "postseason_gap", "postseason", "offseason"]

df["season_phase"] = np.select(conditions, labels, default="unknown")
```

## Workflow

1. Obtain boundary dates for each phase (from a schedule table or hardcoded per year)
2. Define ordered conditions — `np.select` evaluates top-to-bottom, first match wins
3. Assign phase labels as a new categorical column
4. Optionally split further (regular season 1st half / 2nd half, all-star break)
5. Use as a categorical feature in GBDT models or for phase-specific model training

## Key Decisions

- **Condition order**: must be chronological; overlapping conditions resolved by first match
- **Granularity**: 4-6 phases is typical; finer splits (weekly sub-phases) risk overfitting
- **Multi-year**: join a season calendar table by year to handle varying boundary dates
- **Interaction features**: combine phase with day-of-week for richer patterns
- **Phase-specific models**: train separate models per phase when behavior differs drastically

## References

- [EDA of MLB for starter](https://www.kaggle.com/code/chumajin/eda-of-mlb-for-starter-version)
