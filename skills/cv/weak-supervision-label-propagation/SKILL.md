---
name: cv-weak-supervision-label-propagation
description: >
  Propagates image-level multi-labels to individual instances as weak supervision for instance-level training.
---
# Weak Supervision Label Propagation

## Overview

When only image-level labels exist but instance-level predictions are needed, propagate image labels to every detected instance within that image. Each instance inherits all image-level labels, creating weak but usable supervision for training instance classifiers without expensive per-instance annotations.

## Quick Start

```python
import pandas as pd

def propagate_labels(image_labels_df, instances_df):
    """Assign image-level labels to each detected instance."""
    rows = []
    for _, img in image_labels_df.iterrows():
        label_ids = str(img["Label"]).split("|")
        instances = instances_df[instances_df["image_id"] == img["ID"]]
        for _, inst in instances.iterrows():
            for label in label_ids:
                rows.append({
                    "image_id": img["ID"],
                    "instance_id": inst["instance_id"],
                    "label": int(label),
                    "bbox": inst["bbox"],
                    "mask": inst["mask_rle"],
                })
    return pd.DataFrame(rows)
```

## Workflow

1. Run instance segmentation to detect objects/cells per image
2. Load image-level multi-label annotations
3. Duplicate each instance for every label in its parent image
4. Train instance classifier on propagated labels
5. At inference, predict per-instance — model learns to disambiguate despite noisy labels

## Key Decisions

- **Noise tolerance**: Models like neural nets handle label noise well; tree models less so
- **Label cleaning**: Optionally filter with co-occurrence priors or confident learning
- **Instance quality**: Segmentation quality directly impacts downstream accuracy
- **Multi-label vs single-label**: Works best when instances genuinely share image-level labels

## References

- HPA Single Cell Classification competition (Kaggle)
- Source: [mmdetection-for-segmentation-training](https://www.kaggle.com/code/its7171/mmdetection-for-segmentation-training)
