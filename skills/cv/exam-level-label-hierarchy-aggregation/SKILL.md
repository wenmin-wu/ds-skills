---
name: cv-exam-level-label-hierarchy-aggregation
description: Aggregate per-slice predictions into exam-level labels that satisfy a competition's mutual-exclusion hierarchy (positive vs negative vs indeterminate), using a top-down rule cascade — first decide the exam class, then conditionally rescale the dependent labels so the submission stays internally consistent
---

## Overview

Multi-label medical competitions usually impose constraints across labels: a study is `negative_for_pe` XOR `indeterminate` XOR `positive_for_pe`, and per-organ severity labels are only meaningful when the parent label is positive. Per-slice CNNs don't know about these rules and emit independent sigmoid scores that often violate them — `negative_exam=0.7` and `positive_exam=0.6` is contradictory and metric-penalized. The aggregation fix is a top-down cascade: first commit to the exam-level decision based on the strongest evidence (any slice above 0.5 → positive), then rescale the dependent labels conditionally — push winning labels up by `0.5 + score/2`, push losing labels down by `score/2`. The final submission satisfies the hierarchy by construction.

## Quick Start

```python
import numpy as np
import pandas as pd
from scipy.special import softmax

def aggregate_exam(preds, exam_id):
    rows = preds.loc[preds.StudyInstanceUID == exam_id]
    is_positive = (rows.pe_present_on_image >= 0.5).any()

    out = {}
    if is_positive:
        out['negative_exam_for_pe'] = 0
        out['indeterminate']        = rows.indeterminate.min() / 2
    else:
        out['negative_exam_for_pe'] = 1
        if (rows.indeterminate >= 0.5).any():
            out['indeterminate'] = rows.indeterminate.max()
        else:
            out['indeterminate'] = rows.indeterminate.min() / 2

    a, b = rows[['rv_lv_ratio_gte_1', 'rv_lv_ratio_lt_1']].mean().values
    if a > b:
        a, b = a * 2, b / 2
    out['rv_lv_ratio_gte_1'], out['rv_lv_ratio_lt_1'] = softmax([a, b])

    for k in ['leftsided_pe', 'rightsided_pe', 'central_pe']:
        s = rows[k].mean()
        out[k] = (0.5 + s / 2) if is_positive else (s / 2)
    return out
```

## Workflow

1. Group per-slice predictions by exam id (`StudyInstanceUID` or analogous)
2. Decide the top-level exam class from the strongest evidence — `(slice_score >= 0.5).any()` is the standard rule
3. Set mutually exclusive top-level labels deterministically based on the decision
4. For dependent labels (severity, location, etc.), rescale by `0.5 + mean/2` if the parent was positive, `mean/2` if negative — this guarantees they stay below 0.5 in the negative case
5. For paired labels that must softmax to 1.0 (e.g. `rv_lv_ratio_gte_1` vs `lt_1`), apply softmax to the per-exam means after asymmetric pre-amplification of the winner

## Key Decisions

- **Top-down decision first, then rescale**: bottom-up averaging never satisfies the hierarchy.
- **`0.5 + score/2` and `score/2` rescaling**: pushes confident losers below 0.5 and confident winners above 0.5 without losing fine-grained ranking inside each side.
- **`.any()` for positive detection, not `.mean()`**: a single confident positive slice should flip the exam — averaging dilutes it.
- **Asymmetric softmax pre-amplification**: doubling the winner before softmax sharpens the output distribution without distorting the ranking.
- **Persist the rule cascade with the model**: if the metric definition changes, you only update one function.

## References

- [PE Detection with Keras - Model Creation](https://www.kaggle.com/competitions/rsna-str-pulmonary-embolism-detection)
