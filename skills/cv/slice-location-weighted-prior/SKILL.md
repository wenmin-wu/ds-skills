---
name: cv-slice-location-weighted-prior
description: Encode the empirical per-position event rate as a soft prior over slice index by binning normalized slice location and looking up a precomputed weight vector — adds class-aware spatial context to a per-slice classifier without changing the model
---

## Overview

In CT pathology detection, lesion likelihood is rarely uniform across the volume: pulmonary emboli concentrate in specific lung regions, brain hemorrhages cluster by anatomical landmark, kidney stones live in the lower third. A per-slice 2D CNN sees one slice at a time and has no idea where in the volume that slice sits, so it can't exploit this prior. The fix is to compute the empirical event rate per normalized slice-position bin on the training set, store it as a small lookup table, and pass each slice's binned weight as a feature (or multiply it into the predicted score post-hoc). Costs ~10 lines and consistently improves slice-level AUC by a few points.

## Quick Start

```python
import numpy as np
import pandas as pd

# 1. Compute the empirical prior on the training set
N_BINS = 8
df['slice_loc'] = df['series_index'] / (df['num_images'] - 1).clip(lower=1)
df['bin'] = (df['slice_loc'] * N_BINS).clip(0, N_BINS - 1).astype(int)

prior = (
    df.groupby('bin')['has_event']
      .mean()
      .reindex(range(N_BINS), fill_value=0.0)
      .values
)
# e.g. [0.0033, 0.0597, 0.3265, 0.6745, 0.7134, 0.4734, 0.0741, 0.0037]

# 2. At training/inference time, attach the prior as a per-slice feature
def attach_prior(test_df, prior, n_bins=N_BINS):
    test_df['slice_loc'] = test_df['series_index'] / (test_df['num_images'] - 1).clip(lower=1)
    test_df['bin'] = (test_df['slice_loc'] * n_bins).clip(0, n_bins - 1).astype(int)
    test_df['loc_prior'] = test_df['bin'].map(lambda b: prior[b])
    return test_df
```

## Workflow

1. Normalize slice index to `[0, 1]` within each series — series have variable lengths, so absolute index won't transfer
2. Bin the normalized location into N=8 bins (more bins overfit on small training sets)
3. Compute mean event rate per bin on the training set only — never include val/test
4. Attach the per-bin weight as either an extra input feature or a multiplicative post-hoc score adjustment
5. Persist the prior vector with the model checkpoint so inference reuses the same lookup

## Key Decisions

- **8 bins is the sweet spot**: 4 is too coarse to capture lung apex/base asymmetry; 16 overfits per-bin rates on <10k exams.
- **Use mean event rate, not log-odds**: simple mean is interpretable and works as well.
- **Attach as feature, don't only post-multiply**: feature-side lets the model learn an arbitrary nonlinear interaction; post-multiply is a fixed transform that helps less.
- **Compute on training set only**: leak-free — slice positions are not random, so val priors would leak label info.
- **Floor `(num_images - 1)` at 1**: defends against single-slice series that would divide by zero.

## References

- [MONAI 3D CNN - Inference](https://www.kaggle.com/competitions/rsna-str-pulmonary-embolism-detection)
