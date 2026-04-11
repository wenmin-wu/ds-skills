---
name: cv-classifier-gated-detection-postprocess
description: >
  Uses a binary classifier's probability to gate object detector outputs in three tiers: keep detections, append a no-finding box, or replace all detections.
---
# Classifier-Gated Detection Post-Processing

## Overview

In medical imaging detection (chest X-ray, mammography), most images are normal — detectors produce false positives on healthy images. A binary normal/abnormal classifier acts as a gate: if the classifier is confident the image is normal, suppress all detections and output a "No Finding" pseudo-box. If uncertain, append both. This three-tier approach reduces false positives without losing true detections, typically improving mAP by 0.01–0.03.

## Quick Start

```python
import pandas as pd

NORMAL_PRED = "14 1.0 0 0 1 1"  # class_id=14 (No Finding), conf=1.0, full-image box
LOW_THRESH = 0.3   # below: keep detector output only
HIGH_THRESH = 0.7  # above: replace with No Finding

def gate_detections(det_df, clf_df, low_thresh=0.3, high_thresh=0.7):
    """Gate detector predictions using classifier normal-probability."""
    merged = det_df.merge(clf_df[['image_id', 'normal_prob']], on='image_id')
    results = []
    for _, row in merged.iterrows():
        p_normal = row['normal_prob']
        det_str = row['PredictionString']

        if p_normal < low_thresh:
            # Confident abnormal — trust detector
            results.append(det_str)
        elif p_normal < high_thresh:
            # Uncertain — append No Finding alongside detections
            results.append(f"{det_str} {NORMAL_PRED}")
        else:
            # Confident normal — suppress all detections
            results.append(NORMAL_PRED)
    merged['PredictionString'] = results
    return merged

submission = gate_detections(detector_preds, classifier_preds)
```

## Workflow

1. Train a binary classifier (normal vs abnormal) on image-level labels
2. Train an object detector on abnormal images only (or all images)
3. At inference, run both models on each test image
4. Gate detector output based on classifier confidence using two thresholds
5. Tune thresholds on validation set to maximize competition metric

## Key Decisions

- **Threshold tuning**: Grid-search both thresholds on validation mAP; they're dataset-dependent
- **Classifier architecture**: Can be lighter than detector — EfficientNet-B0 is often sufficient
- **No Finding box**: Use full-image bbox `[0, 0, 1, 1]` with high confidence for the normal class
- **vs NMS**: This is complementary to NMS — apply NMS on detections first, then gate

## References

- [VinBigData 2-Class Classifier Complete Pipeline](https://www.kaggle.com/code/corochann/vinbigdata-2-class-classifier-complete-pipeline)
