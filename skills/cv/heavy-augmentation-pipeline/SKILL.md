---
name: cv-heavy-augmentation-pipeline
description: >
  Comprehensive albumentations augmentation combining geometric, photometric, noise, blur, and cutout transforms for robust CV training.
---
# Heavy Augmentation Pipeline

## Overview

Stack diverse augmentations — geometric (rotation, distortion, elastic), photometric (brightness, contrast, hue), degradation (blur, noise, compression), and erasure (cutout) — to force the model to learn invariant features. Use `OneOf` groups to apply one transform per category per sample, keeping the total distortion manageable.

## Quick Start

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(image_size=640):
    return A.Compose([
        A.RandomResizedCrop(image_size, image_size, scale=(0.85, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=60, p=0.5),
        A.OneOf([
            A.OpticalDistortion(distort_limit=1.0),
            A.GridDistortion(num_steps=5),
            A.ElasticTransform(alpha=3),
        ], p=0.2),
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50)),
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MotionBlur(blur_limit=7),
        ], p=0.2),
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.CLAHE(clip_limit=4.0),
        ], p=0.3),
        A.CoarseDropout(max_holes=8, max_height=image_size//10,
                        max_width=image_size//10, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_valid_transforms(image_size=640):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
```

## Workflow

1. Start with spatial: RandomResizedCrop + Flip + ShiftScaleRotate
2. Add distortion group (OneOf): optical, grid, elastic
3. Add degradation group (OneOf): noise, blur
4. Add color group (OneOf): HSV, brightness/contrast, CLAHE
5. Add erasure: CoarseDropout (replaces deprecated Cutout)
6. Always end with Normalize + ToTensorV2

## Key Decisions

- **OneOf groups**: Prevents stacking too many transforms on one sample
- **Probabilities**: Start conservative (0.2-0.3), increase if overfitting persists
- **Medical images**: Skip HueSaturationValue if color is diagnostic (e.g., pathology)
- **CoarseDropout**: Simulates occlusion — critical for detection/classification
- **Validation**: Never augment validation — only resize + normalize

## References

- RANZCR CLiP - Catheter and Line Position Challenge (Kaggle)
- Source: [single-fold-training-of-resnet200d-lb0-965](https://www.kaggle.com/code/underwearfitting/single-fold-training-of-resnet200d-lb0-965)
