---
name: cv-coco-annotation-conversion
description: Convert per-instance RLE or polygon annotations to COCO JSON format for seamless use with Detectron2 and MMDetection
---

# COCO Annotation Conversion

## Overview

Detectron2 and MMDetection expect COCO-format JSON annotations (`images`, `annotations`, `categories` arrays). Competition data often comes as CSV with RLE strings or per-image annotation lists. Converting to COCO JSON enables direct use of `register_coco_instances` and standard data loaders, avoiding custom dataset classes.

## Quick Start

```python
import json
import numpy as np
from pycocotools import mask as mask_util

def build_coco_json(df, image_dir, output_path):
    images, annotations = [], []
    ann_id = 1
    for img_id, (image_name, group) in enumerate(df.groupby('image_id')):
        h, w = group.iloc[0]['height'], group.iloc[0]['width']
        images.append({
            'id': img_id, 'file_name': f'{image_name}.png',
            'height': h, 'width': w
        })
        for _, row in group.iterrows():
            mask = rle_decode(row['annotation'], (h, w))
            rle = mask_util.encode(np.asfortranarray(mask))
            rle['counts'] = rle['counts'].decode('utf-8')
            bbox = mask_util.toBbox(rle).tolist()
            annotations.append({
                'id': ann_id, 'image_id': img_id,
                'category_id': row.get('class_id', 0),
                'segmentation': rle, 'bbox': bbox,
                'bbox_mode': 0, 'area': int(mask.sum()),
                'iscrowd': 0
            })
            ann_id += 1
    coco = {
        'images': images,
        'annotations': annotations,
        'categories': [{'id': 0, 'name': 'cell'}]
    }
    with open(output_path, 'w') as f:
        json.dump(coco, f)

# Register with Detectron2
from detectron2.data.datasets import register_coco_instances
register_coco_instances('train', {}, 'annotations_train.json', 'images/')
```

## Workflow

1. Group annotations by image ID
2. For each annotation: decode mask, re-encode as COCO RLE, compute bbox and area
3. Build the COCO JSON structure with `images`, `annotations`, `categories`
4. Save to disk and register with `register_coco_instances`

## Key Decisions

- **RLE counts as string**: COCO JSON requires `counts` as UTF-8 string, not bytes — decode after encoding
- **bbox_mode**: Detectron2 uses `BoxMode.XYXY_ABS` (mode 0) by default; COCO uses XYWH — check your framework
- **iscrowd**: set to 0 for instance segmentation; 1 for crowd regions to ignore
- **Train/val split**: generate separate JSON files per fold for cross-validation

## References

- [Positive score with Detectron 2/3 - Training](https://www.kaggle.com/code/slawekbiel/positive-score-with-detectron-2-3-training)
- [Sartorius: MMDetection [Train]](https://www.kaggle.com/code/awsaf49/sartorius-mmdetection-train)
