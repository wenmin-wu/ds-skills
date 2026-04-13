---
name: cv-lateralized-label-flip-tta-disable
description: Disable horizontal-flip augmentation (both train-time and TTA) when label columns encode left/right anatomy — flipping silently corrupts the targets because "Left ICA" must map to "Right ICA" after a flip, not stay as "Left ICA"
---

## Overview

Horizontal flip is one of the cheapest TTA tricks and almost always improves classification. But on datasets where labels are *lateralized* — left/right kidney, left/right common carotid, left/right ICA aneurysm — naive `RandomHorizontalFlip` is a silent disaster: the pixels flip but the target string still says "Left X". The model learns a wrong mapping and TTA averages predictions of two opposite labels. There are two valid responses: (A) disable horizontal flip everywhere, or (B) implement a flip+label-swap transform that simultaneously mirrors the image and swaps the corresponding column pairs in the target vector. (A) is simpler and bulletproof; (B) recovers the augmentation budget if you control the transforms tightly. **Default to (A) unless you have label-pair metadata.**

## Quick Start

```python
import albumentations as A

# Option A — safest: just disable horizontal flip
TRAIN_TFM = A.Compose([
    A.Resize(384, 384),
    # NO HorizontalFlip
    A.VerticalFlip(p=0.5),       # safe if no up/down semantics
    A.Rotate(limit=15, p=0.5),
    A.Normalize(),
])
TTA_TRANSFORMS = []  # no flip TTA

# Option B — flip + label swap (only if you have a pair table)
LR_PAIRS = [('Left ICA', 'Right ICA'), ('Left MCA', 'Right MCA'), ...]
def flip_with_label_swap(img, label_vec, cols):
    img = img[:, ::-1, :].copy()
    new = label_vec.copy()
    for l, r in LR_PAIRS:
        new[cols.index(l)], new[cols.index(r)] = label_vec[cols.index(r)], label_vec[cols.index(l)]
    return img, new
```

## Workflow

1. Inspect label columns — any column starting with "Left" / "Right" or "L_" / "R_" is a red flag
2. Decide between option A (disable) and option B (flip+swap with pair table)
3. Remove `HorizontalFlip` from the train transform pipeline (or wrap in a custom class for option B)
4. Remove flip variants from the TTA transform list
5. Add a unit test: run the pipeline on a synthetic image with a left-only label and assert the post-augmentation label is unchanged
6. Document the decision in the data-loading file so future contributors don't reintroduce it

## Key Decisions

- **Default to disable**: the few percent TTA lift isn't worth a hidden label corruption.
- **Vertical flip is usually safe** for axial medical slices but check the dataset: spine images have head/foot semantics.
- **Rotation > 30 degrees can also flip lateralization** in specific projections — keep rotation modest.
- **The bug is silent**: validation metric drops by ~1-3 percentage points but no error is raised. Always check the label space before adding flip.
- **Alternative**: train two heads, one per side, with a single mirrored input — but this is overkill for most pipelines.
- **Generalizes**: any spatial-anatomy label (left/right hand, left/right ear), road-sign datasets (mirror-symmetric arrows), and OCR.

## References

- [RSNA2025 32ch img infer LB 0.69 share](https://www.kaggle.com/code/yamitomo/rsna2025-32ch-img-infer-lb-0-69-share)
