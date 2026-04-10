---
name: tabular-personnel-count-parsing
description: Parse structured text fields like '1 RB, 2 TE, 2 WR' into separate numeric columns per category
domain: tabular
---

# Personnel Count Parsing

## Overview

Datasets often encode categorical counts as comma-separated text (e.g., "1 RB, 2 TE, 2 WR" or "3 DL, 2 LB, 6 DB"). Parse these into separate numeric columns per category. Works for sports rosters, inventory manifests, ingredient lists, or any structured-text count field.

## Quick Start

```python
import pandas as pd

def parse_personnel(text, all_categories=None):
    """Parse '1 RB, 2 TE, 2 WR' into {RB: 1, TE: 2, WR: 2}.
    
    Args:
        text: comma-separated count+category string
        all_categories: list of expected categories (for zero-filling)
    """
    counts = {}
    if all_categories:
        counts = {c: 0 for c in all_categories}
    for item in str(text).split(","):
        item = item.strip()
        parts = item.split(" ")
        if len(parts) == 2:
            counts[parts[1]] = int(parts[0])
    return counts

# Usage
categories = ['DL', 'LB', 'DB', 'OL', 'QB', 'RB', 'TE', 'WR']
parsed = df['OffensePersonnel'].apply(
    lambda x: pd.Series(parse_personnel(x, categories))
)
parsed.columns = [f'offense_{c}' for c in parsed.columns]
df = pd.concat([df, parsed], axis=1)
```

## Key Decisions

- **Zero-fill all categories**: ensures consistent column count even if a category is absent
- **Strip whitespace**: handles inconsistent spacing in raw data
- **Prefix columns**: distinguish offense/defense or source when parsing multiple fields
- **Generalizable**: same pattern for any "count + label" text (e.g., "2x GPU, 1x CPU")

## References

- Source: [location-eda-8eb410](https://www.kaggle.com/code/bestpredict/location-eda-8eb410)
- Competition: NFL Big Data Bowl
