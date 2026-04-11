---
name: nlp-multi-metric-string-similarity
description: >
  Computes multiple complementary string similarity scores (Gestalt, Levenshtein, Jaro-Winkler, LCS) per field pair as features for entity matching classifiers.
---
# Multi-Metric String Similarity

## Overview

No single string similarity metric captures all types of text variation. Gestalt (SequenceMatcher) handles rearrangements, Levenshtein captures edit distance, Jaro-Winkler rewards prefix matches (good for names), and LCS measures shared subsequences. Computing all four per field pair — plus normalized variants — gives a downstream classifier rich signals for entity matching, deduplication, and record linkage.

## Quick Start

```python
import difflib
import Levenshtein

def string_similarity_features(s1, s2):
    """Compute multiple similarity metrics for a string pair."""
    if not s1 or not s2:
        return {"gestalt": -1, "levenshtein": -1, "jaro_winkler": -1,
                "norm_levenshtein": -1}
    gestalt = difflib.SequenceMatcher(None, s1, s2).ratio()
    leven = Levenshtein.distance(s1, s2)
    jaro = Levenshtein.jaro_winkler(s1, s2)
    max_len = max(len(s1), len(s2))
    return {
        "gestalt": gestalt,
        "levenshtein": leven,
        "jaro_winkler": jaro,
        "norm_levenshtein": leven / max_len if max_len > 0 else 0,
    }

# Apply to each field pair in candidate matches
for field in ["name", "address", "city"]:
    feats = df.apply(
        lambda r: string_similarity_features(r[f"{field}_1"], r[f"{field}_2"]),
        axis=1, result_type="expand"
    )
    feats.columns = [f"{field}_{c}" for c in feats.columns]
    df = pd.concat([df, feats], axis=1)
```

## Workflow

1. For each candidate pair, extract corresponding text fields
2. Compute Gestalt ratio, Levenshtein distance, Jaro-Winkler similarity
3. Normalize Levenshtein by max string length for scale-invariant comparison
4. Optionally add LCS length and LCS ratio (LCS / min or max length)
5. Use all metrics as features in a LightGBM/XGBoost classifier

## Key Decisions

- **Which metrics**: Gestalt + Jaro-Winkler + normalized Levenshtein is the minimum useful set
- **Normalization**: Divide edit distance by max length; divide LCS by min length for recall
- **Missing values**: Use -1 sentinel for missing/null fields (tree models handle this well)
- **Performance**: For millions of pairs, use python-Levenshtein (C extension) not pure Python

## References

- [Foursquare - LightGBM Baseline](https://www.kaggle.com/code/ryotayoshinobu/foursquare-lightgbm-baseline)
- [binary_lgb_baseline[0.834]](https://www.kaggle.com/code/guoyonfan/binary-lgb-baseline-0-834)
