---
name: cv-soft-threshold-count-aggregation
description: Aggregate patch-level count-regression predictions into image-level totals by multiplying each prediction by a boolean mask (x * (x > tau)) instead of rounding, retaining fractional evidence above threshold while killing background noise
---

## Overview

Patch count-regression produces noisy fractional outputs everywhere — even empty patches return small positive values that sum to spurious counts over thousands of tiles. Hard rounding throws away real signal on patches with 0.4 objects. The right aggregation is soft thresholding: `preds * (preds > tau)` zeros out any patch below `tau` with a boolean mask but keeps the *full fractional value* for surviving patches, then sums. This is **not** the same as `preds.clip(tau)` or hard rounding — it's a masked sum that preserves the confidence signal.

## Quick Start

```python
import numpy as np

preds = model.predict(test_patches)            # (n_patches, n_classes) float
tau = 0.30                                     # tuned on validation

counts = np.sum(preds * (preds > tau), axis=0).astype('int')
# equivalent, explicit:
# counts = np.where(preds > tau, preds, 0).sum(axis=0).round().astype('int')
```

## Workflow

1. Run patch-level count regression on every test patch
2. Sweep `tau` over `[0.1, 0.2, 0.3, 0.4, 0.5]` on a held-out validation set
3. Pick the `tau` (per-class if needed) minimizing validation RMSE against ground-truth image counts
4. At inference, multiply predictions by `(preds > tau)` and sum across patches
5. Round the result to integer counts for the submission

## Key Decisions

- **Soft mask, not hard round**: `np.round(preds).sum()` discards 0.4-per-patch signal that legitimately sums to objects; soft mask keeps it.
- **Per-class `tau`**: rare classes want lower thresholds to not zero out their weak-but-real signal; dense classes want higher to suppress noise.
- **Tune against RMSE on image-level totals**, not per-patch F1 — the metric you actually ship is the image count.
- **vs. clipping**: `preds.clip(tau)` would floor everything at `tau`, inflating counts. The mask-and-keep is the right operator.
- **Apply after aggregation pooling, not before**: soft threshold only helps on the final summation step; don't apply it inside the model.

## References

- [Use keras to count sea lions](https://www.kaggle.com/code/outrunner/use-keras-to-count-sea-lions)
