---
name: cv-detectron2-custom-data-mapper
description: Custom Detectron2 data mapper with photometric augmentations that properly transforms images, bounding boxes, and instance masks in sync
---

# Detectron2 Custom Data Mapper

## Overview

Detectron2's default data loader applies minimal augmentation. For instance segmentation, you often need photometric augmentations (brightness, contrast, saturation) plus geometric transforms that keep masks and boxes in sync. A custom mapper plugs into `build_detection_train_loader` and applies a chain of `T.Transform` ops that automatically propagate to all annotation types.

## Quick Start

```python
import copy
import torch
from detectron2.data import detection_utils as utils, transforms as T
from detectron2.data import build_detection_train_loader

def custom_mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    transform_list = [
        T.RandomBrightness(0.9, 1.1),
        T.RandomContrast(0.9, 1.1),
        T.RandomSaturation(0.9, 1.1),
        T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
        T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
    ]
    image, transforms = T.apply_transform_gens(transform_list, image)
    dataset_dict["image"] = torch.as_tensor(
        image.transpose(2, 0, 1).astype("float32"))
    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    return dataset_dict

class AugTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper)
```

## Workflow

1. Deep-copy the dataset dict (Detectron2 reuses dicts across epochs)
2. Read image and define a transform chain
3. Apply transforms — `apply_transform_gens` returns the transformed image and the transform object
4. Use `transform_instance_annotations` to apply the same transforms to each annotation
5. Convert to `Instances` and filter empty ones
6. Override `build_train_loader` in a custom Trainer subclass

## Key Decisions

- **Deep copy**: mandatory — without it, augmentations corrupt the original dataset dict
- **iscrowd filter**: skip crowd annotations that break instance-level evaluation
- **filter_empty_instances**: removes annotations with zero-area masks after cropping/flipping
- **Geometric augments**: add `T.ResizeShortestEdge`, `T.RandomCrop` for scale variation

## References

- [Sartorius Segmentation - Detectron2 [Training]](https://www.kaggle.com/code/ammarnassanalhajali/sartorius-segmentation-detectron2-training)
