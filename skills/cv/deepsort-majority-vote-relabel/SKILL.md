---
name: cv-deepsort-majority-vote-relabel
description: Run DeepSort on per-frame detections then overwrite each track cluster's label with the most common mapped label across the track's lifetime
---

## Overview

DeepSort produces temporally-consistent tracker IDs, but the downstream *labels* assigned to each detection frame-by-frame are still noisy. The fix: after DeepSort assigns a stable `cluster_id`, look up all per-frame labels inside that cluster and overwrite every frame in the cluster with the majority-vote label. One wrong frame cannot flip a whole track, and flickering ID assignments collapse into one canonical ID per player/object. Used in the NFL Helmet Assignment competition to lift assignment accuracy by 5-10 points.

## Quick Start

```python
from deep_sort_pytorch.deep_sort import DeepSort
import cv2, pandas as pd

deepsort = DeepSort(ckpt, max_dist=0.2, max_iou_distance=0.9,
                    max_age=15, n_init=1, nn_budget=30, use_cuda=True)

tracked = []
for frame, d in detections.groupby('frame'):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame - 1)
    _, image = cap.read()
    xywhs = d[['x','y','width','height']].values
    out = deepsort.update(xywhs, confs=np.ones(len(d)),
                          clss=np.zeros(len(d)), ori_img=image)
    df_out = pd.DataFrame(out, columns=['l','t','r','b','cluster','cls'])
    # Merge cluster ids back onto the original label column
    d = pd.merge_asof(d.sort_values('x'), df_out.sort_values('l'),
                      left_on='x', right_on='l', direction='nearest')
    tracked.append(d)

tracked = pd.concat(tracked)

# Majority vote per cluster
vote = (tracked.groupby('cluster')['label']
        .agg(lambda s: s.value_counts().idxmax()))
tracked['label'] = tracked['cluster'].map(vote)

# Drop duplicate labels per frame (submission constraint)
tracked = tracked.loc[~tracked[['frame','label']].duplicated()]
```

## Workflow

1. Feed per-frame detections into DeepSort to get `cluster_id` for every box
2. Join the original label / class column onto the tracked boxes by nearest spatial match
3. Group by `cluster_id` and compute the modal label of each cluster
4. Overwrite every frame's label with the modal label
5. Drop duplicate (frame, label) pairs — multiple clusters may collapse onto the same label

## Key Decisions

- **Majority vote over per-frame**: single-frame errors (occlusion, similar appearance) flip back to the cluster consensus.
- **Tune `max_age`**: too short and tracks fragment; too long and different objects merge. For sports video, 14-30 frames is typical.
- **Tune `nn_budget`**: controls the re-ID feature gallery size per track. 30-100 works; larger slows inference.
- **vs. per-frame argmax**: pure classifier argmax has no temporal prior and flickers constantly; the majority-vote post-process is nearly free and almost always better.

## References

- [Tuning DeepSort + Helmet Mapping](https://www.kaggle.com/code/its7171/tuning-deepsort-helmet-mapping-high-score)
- [Helper Code + Helmet Mapping + Deepsort](https://www.kaggle.com/code/robikscube/helper-code-helmet-mapping-deepsort)
