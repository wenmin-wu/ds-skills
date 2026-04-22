---
name: timeseries-density-to-count-roundtrip
description: Convert a density metric back to integer counts using known population, round to nearest integer, then recompute density to exploit the discrete nature of the target
---

# Density-to-Count Roundtrip

## Overview

When the target variable is a ratio (count / population), the true underlying quantity is an integer count. Predictions that ignore this produce values on a continuous scale that can never exactly match the true density. Converting predicted density to integer count via the known population denominator, rounding, and converting back snaps predictions to the set of possible true values — often improving SMAPE by 0.001-0.01.

## Quick Start

```python
import numpy as np

def roundtrip(predicted_density, population):
    count = predicted_density * population / 100  # density is per 100 people
    count_rounded = np.round(count).astype(int)
    count_rounded = np.maximum(count_rounded, 0)
    return count_rounded / population * 100

df["pred_snapped"] = roundtrip(df["pred_density"], df["population"])
```

## Workflow

1. Obtain the population denominator for each entity (e.g., from Census ACS data by FIPS code)
2. Convert predicted density to raw count: `count = density * population / scale_factor`
3. Round count to the nearest non-negative integer
4. Convert back: `snapped_density = rounded_count / population * scale_factor`
5. Use snapped density as the final submission

## Key Decisions

- **Population source**: must match the denominator used to compute the original density — mismatched population introduces systematic bias
- **Scale factor**: check whether density is per 100, per 1000, or per capita
- **Rounding**: `round()` (nearest) outperforms `floor()` or `ceil()` in expectation
- **When to apply**: always apply as a post-processing step — it's free improvement when the target is genuinely discrete
- **Population updates**: if population changes over time (e.g., yearly Census updates), use the correct year's population for each prediction period

## References

- [Use Discrete Nature of microbusiness_density](https://www.kaggle.com/code/vitalykudelya/use-discrete-nature-of-microbusiness-density)
