---
name: cv-two-stage-classify-then-detect
description: Chain a study-level classifier with an image-level detector, merging class probabilities and bounding boxes into a unified prediction
domain: cv
---

# Two-Stage Classify-then-Detect Pipeline

## Overview

When predictions span multiple granularities (e.g., study-level disease type + image-level lesion boxes), use a two-stage pipeline: first classify the overall case, then detect specific regions. Merge both outputs into a unified submission. Common in medical imaging where diagnosis (classification) and localization (detection) are both required.

## Quick Start

```python
import numpy as np

# Stage 1: Study-level classification (e.g., EfficientNet)
study_preds = np.zeros((len(studies), n_classes))
for model in fold_models:
    study_preds += model.predict(study_dataset)
study_preds /= len(fold_models)

# Stage 2: Image-level detection (e.g., YOLOv5, Cascade RCNN)
image_detections = {}
for img_id, img_path in images:
    boxes, scores = detector.predict(img_path)
    image_detections[img_id] = (boxes, scores)

# Merge: study gets class probabilities, images get bounding boxes
results = []
for study_id, preds in zip(study_ids, study_preds):
    pred_str = " ".join(
        f"{cls} {prob:.4f} 0 0 1 1" for cls, prob in zip(class_names, preds)
    )
    results.append({"id": f"{study_id}_study", "prediction": pred_str})
    for img_id in study_images[study_id]:
        boxes, scores = image_detections.get(img_id, ([], []))
        if boxes:
            pred_str = " ".join(
                f"opacity {s:.4f} {b[0]} {b[1]} {b[2]} {b[3]}"
                for b, s in zip(boxes, scores)
            )
        else:
            pred_str = "none 1 0 0 1 1"
        results.append({"id": f"{img_id}_image", "prediction": pred_str})
```

## Key Decisions

- **Separate models**: classifier and detector trained independently — avoids task interference
- **Fold averaging on classifier**: reduces variance on study-level predictions
- **Confidence passthrough**: detector confidence scores propagate directly to submission
- **Default fallback**: images with no detections get a "none" prediction with full confidence

## References

- Source: [siim-cov19-efnb7-yolov5-infer](https://www.kaggle.com/code/h053473666/siim-cov19-efnb7-yolov5-infer)
- Competition: SIIM-FISABIO-RSNA COVID-19 Detection
