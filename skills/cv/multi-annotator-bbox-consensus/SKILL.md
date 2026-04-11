---
name: cv-multi-annotator-bbox-consensus
description: >
  Merges overlapping same-class bounding boxes from multiple annotators into a consensus box using IoU-based matching and intersection.
---
# Multi-Annotator Bbox Consensus

## Overview

Datasets annotated by multiple experts (radiologists, pathologists) contain duplicate and conflicting bounding boxes for the same object. Using all raw annotations introduces noise; using only one annotator discards information. This technique groups overlapping same-class boxes by IoU, then computes a consensus box — either the intersection (inner box, conservative) or union (outer box, inclusive). Reduces annotation noise and improves detector training, especially when annotators have varying skill levels.

## Quick Start

```python
import numpy as np

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 < x1 or y2 < y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    return inter / (area1 + area2 - inter)

def consensus_box(boxes, mode='inner'):
    """Compute consensus from overlapping boxes."""
    boxes = np.array(boxes)
    if mode == 'inner':  # intersection
        return [boxes[:,0].max(), boxes[:,1].max(),
                boxes[:,2].min(), boxes[:,3].min()]
    else:  # union
        return [boxes[:,0].min(), boxes[:,1].min(),
                boxes[:,2].max(), boxes[:,3].max()]

def merge_annotations(annots, iou_thresh=0.0, mode='inner'):
    """Merge same-class overlapping boxes from multiple annotators."""
    merged = []
    used = set()
    for i, (cls_i, box_i) in enumerate(annots):
        if i in used:
            continue
        group = [box_i]
        for j, (cls_j, box_j) in enumerate(annots):
            if j <= i or j in used or cls_i != cls_j:
                continue
            if compute_iou(box_i, box_j) > iou_thresh:
                group.append(box_j)
                used.add(j)
        merged.append((cls_i, consensus_box(group, mode)))
    return merged
```

## Workflow

1. Group annotations by image_id
2. For each same-class pair, compute IoU
3. Cluster overlapping boxes (IoU > threshold)
4. Compute consensus box per cluster (inner or outer)
5. Use consensus annotations for training

## Key Decisions

- **Inner vs outer**: Inner (intersection) is conservative, reduces box size; outer (union) is inclusive
- **IoU threshold**: 0.0 merges any overlapping boxes; 0.3–0.5 requires significant overlap
- **Minimum annotators**: Optionally require 2+ annotators to agree before keeping a box
- **Weighted average**: Use annotator-weighted mean of coordinates instead of hard intersection

## References

- [Visual In-Depth EDA – VinBigData](https://www.kaggle.com/code/dschettler8845/visual-in-depth-eda-vinbigdata-competition-data)
