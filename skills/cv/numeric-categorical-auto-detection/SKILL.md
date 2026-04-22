---
name: cv-numeric-categorical-auto-detection
description: Auto-detect whether a generated data series is numeric or categorical by measuring the fraction of digit characters in the concatenated values
---

# Numeric-Categorical Auto Detection

## Overview

When a generative model produces data series from charts, the output is always strings — but downstream processing differs for numeric values (float parsing, RMSE scoring) vs categorical values (string matching, Levenshtein scoring). Auto-detect the data type by computing the digit-character fraction of the concatenated series. If ≥50% of characters are digits, treat as numeric and parse; otherwise treat as categorical strings.

## Quick Start

```python
import re

def detect_series_type(values):
    all_chars = "".join(values)
    if len(all_chars) == 0:
        return "categorical"
    digit_chars = len(re.sub(r"[^\d]", "", all_chars))
    frac_numeric = digit_chars / len(all_chars)
    return "numeric" if frac_numeric >= 0.5 else "categorical"

def process_series(values):
    if detect_series_type(values) == "numeric":
        return clean_numeric(values)  # parse to float/int
    else:
        return [s.strip() for s in values]  # keep as strings

x_series = ["2020", "2021", "2022"]
y_series = ["Apple", "Banana", "Cherry"]
detect_series_type(x_series)  # "numeric"
detect_series_type(y_series)  # "categorical"
```

## Workflow

1. Concatenate all values in the series into a single string
2. Count digit characters using `re.sub(r"[^\d]", "", text)`
3. Compute fraction: digit_count / total_length
4. If fraction ≥ 0.5 → numeric (parse with float/int)
5. If fraction < 0.5 → categorical (keep as stripped strings)

## Key Decisions

- **0.5 threshold**: balanced cutoff; numeric series like "12.5" have ~80% digits, labels like "Q1 2023" have ~40%
- **Concatenated check**: more robust than per-value — a single "N/A" in a numeric series won't flip the type
- **Digit-only count**: counts 0-9 only — decimals, signs, and spaces don't count as "numeric evidence"
- **Fallback**: empty series defaults to categorical (safer for downstream processing)

## References

- [donut-infer (LB 0.44) [benetech]](https://www.kaggle.com/code/nbroad/donut-infer-lb-0-44-benetech)
- [Tuned Donut](https://www.kaggle.com/code/cody11null/tuned-donut)
