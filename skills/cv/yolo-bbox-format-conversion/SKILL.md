---
name: cv-yolo-bbox-format-conversion
description: Convert bounding boxes between YOLO (normalized center), VOC (absolute corners), and COCO (absolute xywh) formats with image dimension scaling
domain: cv
---

# YOLO Bbox Format Conversion

## Overview

Object detection frameworks use different bbox formats: YOLO (normalized x_center, y_center, w, h), VOC/Pascal (absolute xmin, ymin, xmax, ymax), COCO (absolute x, y, w, h). Converting between them is error-prone — off-by-one in normalization or axis order silently degrades mAP. Keep these converters as tested utilities.

## Quick Start

```python
import numpy as np

def voc_to_yolo(boxes, img_w, img_h):
    """VOC [xmin,ymin,xmax,ymax] → YOLO [cx,cy,w,h] normalized."""
    boxes = np.array(boxes, dtype=np.float64)
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    cx = boxes[:, 0] + w / 2
    cy = boxes[:, 1] + h / 2
    return np.stack([cx / img_w, cy / img_h, w / img_w, h / img_h], axis=1)

def yolo_to_voc(boxes, img_w, img_h):
    """YOLO [cx,cy,w,h] normalized → VOC [xmin,ymin,xmax,ymax] absolute."""
    boxes = np.array(boxes, dtype=np.float64)
    boxes[:, [0, 2]] *= img_w
    boxes[:, [1, 3]] *= img_h
    xmin = boxes[:, 0] - boxes[:, 2] / 2
    ymin = boxes[:, 1] - boxes[:, 3] / 2
    xmax = boxes[:, 0] + boxes[:, 2] / 2
    ymax = boxes[:, 1] + boxes[:, 3] / 2
    return np.stack([xmin, ymin, xmax, ymax], axis=1)

def scale_boxes(boxes, from_size, to_size):
    """Rescale absolute boxes from one image size to another."""
    boxes = np.array(boxes, dtype=np.float64)
    sx = to_size[1] / from_size[1]  # width ratio
    sy = to_size[0] / from_size[0]  # height ratio
    boxes[:, [0, 2]] *= sx
    boxes[:, [1, 3]] *= sy
    return boxes
```

## Key Decisions

- **Float64 precision**: avoid rounding errors in normalized coordinates
- **Batch vectorized**: numpy operations over loops for speed
- **Scale separately**: resize image dimensions independently — aspect ratio may change
- **Clip to bounds**: after conversion, `np.clip(boxes, 0, [W,H,W,H])` prevents out-of-frame boxes

## References

- Source: [train-covid-19-detection-using-yolov5](https://www.kaggle.com/code/ayuraj/train-covid-19-detection-using-yolov5)
- Competition: SIIM-FISABIO-RSNA COVID-19 Detection
