---
name: cv-multi-series-channel-stacking
description: >
  Stacks uniformly sampled slices from multiple MRI series (e.g., Sagittal T1, T2, Axial) into a single multi-channel tensor for one-pass inference.
---
# Multi-Series Channel Stacking

## Overview

Medical imaging studies often contain multiple series (Sagittal T1, Sagittal T2/STIR, Axial T2) that provide complementary information. Instead of running separate models, stack a fixed number of uniformly sampled slices from each series into one multi-channel input tensor. A 3-series setup with 10 slices each produces a 30-channel input — any CNN backbone accepts this via `in_chans` override. This fuses cross-series information at the input level, letting the model learn which modality matters for each condition.

## Quick Start

```python
import numpy as np
import pydicom

def sample_slices(dicom_paths, n_slices=10):
    """Uniformly sample n_slices from center of a DICOM series."""
    total = len(dicom_paths)
    step = total / n_slices
    start = total / 2.0 - (n_slices / 2.0 - 0.5) * step
    indices = [max(0, int(round(start + i * step))) for i in range(n_slices)]
    indices = [min(i, total - 1) for i in indices]
    return [dicom_paths[i] for i in indices]

def build_multichannel(series_dict, img_size=256, n_per_series=10):
    """Stack slices from multiple series into (H, W, C) array."""
    series_keys = sorted(series_dict.keys())  # deterministic order
    total_channels = len(series_keys) * n_per_series
    x = np.zeros((img_size, img_size, total_channels), dtype=np.uint8)

    for idx, key in enumerate(series_keys):
        paths = sorted(series_dict[key])
        sampled = sample_slices(paths, n_per_series)
        for j, path in enumerate(sampled):
            img = pydicom.dcmread(path).pixel_array
            img = cv2.resize(img, (img_size, img_size))
            img = np.clip(img / img.max() * 255, 0, 255) if img.max() > 0 else img
            x[..., idx * n_per_series + j] = img.astype(np.uint8)
    return x
```

## Workflow

1. Group DICOM files by series description (Sagittal T1, Axial T2, etc.)
2. Sort slices within each series by instance number
3. Uniformly sample N slices from the center of each series
4. Stack into a single `(H, W, N_series × N_slices)` tensor
5. Feed to any timm model with `in_chans=total_channels`

## Key Decisions

- **Slices per series**: 10 is common; more captures detail but increases memory
- **Sampling strategy**: Center-biased uniform sampling avoids edge slices with less anatomy
- **Missing series**: Fill with zeros — the model learns to ignore empty channels
- **Normalization**: Min-max per slice to [0, 255] uint8 before stacking

## References

- [RSNA2024 LSDC DenseNet Submission](https://www.kaggle.com/code/hugowjd/rsna2024-lsdc-densenet-submission)
- [RSNA2024 LSDC Submission Baseline](https://www.kaggle.com/code/itsuki9180/rsna2024-lsdc-submission-baseline)
