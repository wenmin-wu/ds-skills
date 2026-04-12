---
name: cv-iou-weighted-assignment-metric
description: Evaluation scorer that merges predictions with GT per frame, takes top-IoU match per GT, and computes weighted accuracy with IoU threshold gate
---

## Overview

For multi-object ID assignment problems (NFL helmets, pedestrians, cells), the natural metric is "did the predicted ID match the GT ID, for the correct box?". Counting IoU alone misses identity errors, counting ID alone misses localization errors. The combined metric: for each GT box per frame, find the predicted box with the highest IoU, gate by an IoU threshold (e.g. 0.35), check ID equality, then compute weighted accuracy where high-importance rows (impact plays, critical events) get a 1000× weight. The vectorized IoU computation makes it fast enough to run on every training step.

## Quick Start

```python
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

def vectorized_iou(df):
    ixmin = df[['x1_sub','x1_gt']].max(axis=1)
    iymin = df[['y1_sub','y1_gt']].max(axis=1)
    ixmax = df[['x2_sub','x2_gt']].min(axis=1)
    iymax = df[['y2_sub','y2_gt']].min(axis=1)
    iw = np.maximum(ixmax - ixmin + 1, 0)
    ih = np.maximum(iymax - iymin + 1, 0)
    inter = iw * ih
    area_sub = (df['x2_sub']-df['x1_sub']+1)*(df['y2_sub']-df['y1_sub']+1)
    area_gt  = (df['x2_gt' ]-df['x1_gt' ]+1)*(df['y2_gt' ]-df['y1_gt' ]+1)
    df['iou'] = inter / (area_sub + area_gt - inter)
    return df

def score(sub, gt, iou_thr=0.35, impact_weight=1000):
    merged = gt.merge(sub, on='video_frame', suffixes=('_gt','_sub'))
    merged = vectorized_iou(merged)
    top = (merged.sort_values('iou', ascending=False)
           .groupby(['video_frame','label_gt']).first().reset_index())
    top['correct'] = (top['label_gt'] == top['label_sub']) & (top['iou'] >= iou_thr)
    top['weight']  = np.where(top['isImpact'], impact_weight, 1)
    return accuracy_score(np.ones(len(top)), top['correct'],
                          sample_weight=top['weight'])
```

## Workflow

1. Merge predictions and GT on the common frame/time key
2. Compute IoU vectorized across the merged dataframe (no Python loops)
3. Sort by IoU descending and take the top row per `(frame, gt_label)` — this is the GT's best match
4. Flag `correct = (label matches) AND (iou ≥ threshold)`
5. Apply per-row importance weights and compute weighted accuracy

## Key Decisions

- **Per-GT top-match, not per-pred**: ensures every GT is scored exactly once, even if multiple preds overlap it.
- **IoU gate vs. soft-IoU**: a hard threshold matches the competition rubric and is interpretable; soft-IoU rewards near-misses but is harder to tune.
- **Importance weights**: impact plays / critical frames get 1000× weight — mirrors the downstream cost of errors.
- **vs. per-class F1**: assignment problems usually care about identity preservation, not class recall; this metric is simpler and more stable.

## References

- [NFL Helmet Assignment - Getting Started Guide](https://www.kaggle.com/code/robikscube/nfl-helmet-assignment-getting-started-guide)
