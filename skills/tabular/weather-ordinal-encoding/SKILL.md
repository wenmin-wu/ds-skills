---
name: tabular-weather-ordinal-encoding
description: Map free-text categorical descriptions to ordinal numeric scores via keyword matching — captures ordered severity in a single dense feature
domain: tabular
---

# Weather Ordinal Encoding

## Overview

Free-text categorical fields with inherent ordering (weather, severity, quality) can be encoded as a single ordinal score via keyword matching. Map keywords to numeric values that reflect the ordering (e.g., sunny=2, cloudy=−1, rain=−2). Handles typos and compound descriptions (e.g., "partly cloudy") via multiplicative modifiers.

## Quick Start

```python
import pandas as pd
import numpy as np

def encode_ordinal_text(text, keyword_scores, modifiers=None):
    """Map free-text to ordinal score via keyword matching.
    
    Args:
        text: raw text string
        keyword_scores: dict of keyword → score, checked in order
        modifiers: dict of keyword → multiplier (e.g., 'partly' → 0.5)
    """
    if pd.isna(text):
        return 0
    text = text.lower().strip()
    score = 1.0
    if modifiers:
        for kw, mult in modifiers.items():
            if kw in text:
                score *= mult
    for kw, val in keyword_scores.items():
        if kw in text:
            return score * val
    return 0

# Weather encoding
weather_scores = {
    'indoor': 3, 'climate controlled': 3,
    'sunny': 2, 'sun': 2, 'clear': 1,
    'cloudy': -1, 'overcast': -1,
    'rain': -2, 'rainy': -2, 'showers': -2,
    'snow': -3, 'blizzard': -3,
}
modifiers = {'partly': 0.5, 'light': 0.5}

df['weather_score'] = df['GameWeather'].apply(
    lambda x: encode_ordinal_text(x, weather_scores, modifiers)
)
```

## Key Decisions

- **Single feature**: one dense column instead of N one-hot columns — preserves ordinal relationship
- **Keyword priority**: check in order; first match wins — put specific terms before general ones
- **Modifiers stack**: "partly cloudy" = 0.5 × (−1) = −0.5; handles compound descriptions
- **NaN → 0**: neutral default for missing values; alternatively use median imputation

## References

- Source: [location-eda-8eb410](https://www.kaggle.com/code/bestpredict/location-eda-8eb410)
- Competition: NFL Big Data Bowl
