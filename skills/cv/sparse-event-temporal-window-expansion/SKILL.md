---
name: cv-sparse-event-temporal-window-expansion
description: Turn sparse single-frame event labels (impacts, collisions, goals) into a usable detector training set by stamping the positive label onto a ±k-frame window around each event, then dropping any frame that contains no positives — gives the detector enough positive samples without changing the original annotation
---

## Overview

Video event datasets are brutal for object detectors: a "helmet impact" might be marked on exactly one frame in a 600-frame clip, leaving a 1:600 positive ratio that no detector can train on. Worse, the labeled frame may not even be the visually clearest one — events span 4-8 frames in reality. The fix is temporal window expansion: for every positive frame, mark frames `t-k…t+k` (typically k=4) as positive too, using the same bounding box. Then filter the dataset to keep only frames that contain at least one positive — discarding the boring 590 negative frames per clip. The detector now sees ~9x more positive samples without you having to relabel anything.

## Quick Start

```python
import numpy as np
import pandas as pd

video_labels = pd.read_csv('train_labels.csv').fillna(0)
positives = video_labels[video_labels['impact'] > 0]

K = 4   # +/- frames around each event
offsets = np.array([-4, -3, -2, -1, 1, 2, 3, 4])

for vid, frame, lbl in positives[['video', 'frame', 'label']].values:
    nbr_frames = offsets + frame
    mask = (
        (video_labels['video']  == vid) &
        (video_labels['frame'].isin(nbr_frames)) &
        (video_labels['label']  == lbl)
    )
    video_labels.loc[mask, 'impact'] = 1

# drop frames with no positives at all
video_labels = video_labels[
    video_labels.groupby('image_name')['impact'].transform('sum') > 0
].reset_index(drop=True)

# remap to 1=non-impact, 2=impact for 2-class detector
video_labels['impact'] = video_labels['impact'].astype(int) + 1
```

## Workflow

1. Identify rows with positive event labels in the original annotations
2. For each positive `(video, frame, label)`, mark frames in `[frame-k, frame+k]` of the same `(video, label)` as positive too — match the bounding box of the existing tracked object
3. Group by `image_name` and drop any frame whose total positive count is zero
4. (Optional) Remap labels so the detector learns "background class" vs "event class" instead of binary
5. Cache the expanded label CSV — never expand on the fly during training, it's slow and non-deterministic

## Key Decisions

- **k=4 for 60fps video**: covers ~130ms of event duration, enough to capture impact buildup and follow-through; smaller k starves the detector, larger k mislabels rest frames.
- **Keep the same bbox**: the tracked object hasn't moved much in 4 frames; reusing the box is more accurate than interpolation.
- **Drop negative-only frames**: keeping them adds noise without signal — the negatives within positive frames are already plenty.
- **Same-label expansion only**: don't bleed a "helmet impact" label onto a "shoulder impact" track in the same neighborhood.
- **vs. label smoothing**: label smoothing softens the loss but doesn't change the positive rate; window expansion fixes both.

## References

- [2Class Object Detection Training](https://www.kaggle.com/competitions/nfl-impact-detection)
