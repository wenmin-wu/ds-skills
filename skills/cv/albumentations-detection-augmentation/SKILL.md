---
name: cv-albumentations-detection-augmentation
description: >
  Applies albumentations augmentations to object detection data while preserving bbox-label correspondence via BboxParams.
---
# Albumentations Detection Augmentation

## Overview

Albumentations supports spatial transforms (flip, rotate, crop, resize) that automatically adjust bounding boxes alongside the image. The key is `BboxParams` — it tells the pipeline the bbox format (pascal_voc, coco, yolo) and which field maps labels to boxes. Without this, augmented bboxes get shuffled or lost. This pattern wraps albumentations into a detection framework's data mapper (Detectron2, MMDetection) or a custom PyTorch Dataset.

## Quick Start

```python
import albumentations as A
import numpy as np

# Define augmentation pipeline with bbox support
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                        rotate_limit=15, p=0.5),
    A.RandomResizedCrop(height=512, width=512, scale=(0.8, 1.0), p=0.5),
], bbox_params=A.BboxParams(
    format='pascal_voc',          # [x_min, y_min, x_max, y_max]
    label_fields=['category_ids'],
    min_area=100,                 # drop tiny boxes after crop
    min_visibility=0.3,           # drop mostly-clipped boxes
))

# Apply to image + bboxes
bboxes = [[100, 100, 300, 300], [400, 200, 500, 400]]
labels = [0, 1]

result = transform(
    image=image,
    bboxes=bboxes,
    category_ids=labels,
)
aug_image = result['image']
aug_bboxes = result['bboxes']       # transformed coordinates
aug_labels = result['category_ids']  # preserved correspondence
```

## Workflow

1. Define augmentation pipeline with `A.Compose` and `A.BboxParams`
2. Specify bbox format: `pascal_voc`, `coco` (x,y,w,h), or `yolo` (normalized)
3. Map label fields so they track with bboxes through transforms
4. Set `min_area` and `min_visibility` to drop degenerate boxes after cropping
5. Apply in Dataset `__getitem__` or framework-specific mapper

## Key Decisions

- **Format**: Match your annotation format — `pascal_voc` for [x1,y1,x2,y2], `yolo` for normalized [cx,cy,w,h]
- **min_area/min_visibility**: Prevents training on tiny slivers; 100px² area and 0.3 visibility are safe defaults
- **Spatial only**: Non-spatial transforms (brightness, contrast) don't affect bboxes — safe to add freely
- **Label tracking**: `label_fields` links labels to bboxes; without it, augmented boxes lose their class IDs

## References

- [VinBigData Detectron2 Train](https://www.kaggle.com/code/corochann/vinbigdata-detectron2-train)
