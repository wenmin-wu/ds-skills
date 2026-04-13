---
name: cv-modality-adaptive-dicom-windowing
description: Route each DICOM series to a per-modality window-center / window-width pair (CT/CTA/MRA/MRI) before normalization, so the same model can ingest mixed modalities without one modality's intensity range washing out the others
---

## Overview

Mixed-modality medical datasets like RSNA Intracranial Aneurysm contain CT, CTA, MRA, and MRI series in the same training set. A single fixed windowing (or per-image min-max) collapses contrast on whichever modality wasn't tuned for. The fix: read `Modality` from the DICOM header, look it up in a `{modality: (window_center, window_width)}` table, and apply the matching window before scaling to uint8. The result is that vessel structure on CTA, flow signal on MRA, and parenchyma on MRI each land in a comparable normalized range — and one shared model can treat all of them as a "vessel image" rather than learning modality-specific intensity hacks.

## Quick Start

```python
import numpy as np
import pydicom

WINDOWS = {
    'CT':  (40,   80),     # parenchyma
    'CTA': (50,   350),    # vessels
    'MRA': (600,  1200),   # bright-blood angio
    'MRI': (40,   80),     # generic MR
}

def window(img, modality):
    wc, ww = WINDOWS.get(modality, (40, 80))
    lo, hi = wc - ww // 2, wc + ww // 2
    img = np.clip(img, lo, hi)
    return ((img - lo) / (hi - lo + 1e-7) * 255).astype(np.uint8)

ds = pydicom.dcmread(fp, force=True)
px = ds.pixel_array.astype(np.float32) * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
img = window(px, getattr(ds, 'Modality', 'CT'))
```

## Workflow

1. Inventory unique `Modality` strings in your training set (`CT`, `CTA`, `MR`, `MRA`)
2. Look up clinically-correct window center/width per modality (radiology atlases or kaggle host docs)
3. Always apply `RescaleSlope * px + RescaleIntercept` first — CT only, but harmless on MR
4. Window then scale to `[0, 255]` uint8 (or `[0, 1]` float)
5. Pass through the same downstream loader/augmentation as if all modalities were a single "image"
6. As a feature, also keep a one-hot modality embedding fed to the head — gives the model an explicit signal it cares about per-modality differences

## Key Decisions

- **Window first, normalize second**: per-image min-max on raw HU is dominated by air/bone outliers and destroys soft-tissue contrast.
- **Rescale slope/intercept before windowing**: stored pixel values are not HU until rescaled.
- **MR has no HU**: clip to per-volume percentiles (1, 99) instead of fixed numbers if the host hasn't provided a window.
- **Per-modality vs. multi-window stack**: single window per modality is simpler; if you have GPU budget, stack 3 windows per modality for a richer input.
- **Always carry the modality string forward**: it's a strong feature for any classification head.
- **Hardcoded windows beat data-driven**: clinical windows are robust; learned ones overfit per-fold.

## References

- [RSNA-IAD | Ensemble | LB #1](https://www.kaggle.com/code/zhouzhi73/rsna-iad-ensemble-lb-1)
