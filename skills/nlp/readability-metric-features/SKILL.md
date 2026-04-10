---
name: nlp-readability-metric-features
description: >
  Extracts established readability scores (Flesch, Gunning FOG, ARI, Coleman-Liau) as numeric features from text.
---
# Readability Metric Features

## Overview

Established readability formulas compute text complexity from sentence length, syllable counts, and word difficulty. Extract these as features for regression or classification tasks involving text quality, difficulty, or grade level. The `textstat` library provides a unified API for all major formulas.

## Quick Start

```python
import textstat

def extract_readability(text):
    return {
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
        "gunning_fog": textstat.gunning_fog(text),
        "ari": textstat.automated_readability_index(text),
        "coleman_liau": textstat.coleman_liau_index(text),
        "dale_chall": textstat.dale_chall_readability_score(text),
        "linsear_write": textstat.linsear_write_formula(text),
        "text_standard": textstat.text_standard(text, float_output=True),
    }

df["readability"] = df["text"].apply(extract_readability)
features = pd.json_normalize(df["readability"])
```

## Workflow

1. Install `textstat` (`pip install textstat`)
2. Apply each readability formula to every text sample
3. Combine scores into a feature dataframe
4. Use as standalone features for GBDT or as auxiliary features alongside embeddings

## Key Decisions

- **Which formulas**: Flesch + Gunning FOG + ARI cover most variance; others add marginal value
- **Correlation**: Many formulas are correlated — use with tree models (handle collinearity)
- **Complement embeddings**: These capture surface complexity; embeddings capture semantics
- **Short texts**: Readability formulas are unreliable below ~50 words

## References

- CommonLit Readability Prize (Kaggle)
- Source: [commonlit-readability-prize-eda-baseline](https://www.kaggle.com/code/ruchi798/commonlit-readability-prize-eda-baseline)
