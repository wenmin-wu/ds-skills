---
name: cv-bbox-augmentation-retry-loop
description: Wrap albumentations bbox transforms in a 10-attempt retry loop that re-rolls the augmentation when all boxes get cropped out, preventing empty-target samples from poisoning the detection loss with NaN gradients
---

## Overview

Albumentations' bbox-aware transforms (`RandomCrop`, `RandomSizedBBoxSafeCrop`, big rotations) silently drop bounding boxes that fall outside the new canvas. For dense scenes this is fine, but for sparse-label data — one or two boxes per image — a single unlucky crop wipes the entire ground truth and you feed an "image with no targets" into a detector that expects at least one box. EfficientDet, FasterRCNN, and YOLO all behave badly on empty targets: silent NaN losses, KL collapse, or skipped batches that bias the optimizer. The fix is a 10-iteration retry loop: rerun the transform until at least one box survives, fall through to the un-augmented sample as a last resort.

## Quick Start

```python
def __getitem__(self, idx):
    image, target, labels = self._load(idx)

    if self.transforms is not None:
        for _ in range(10):                       # up to 10 attempts
            sample = self.transforms(
                image=image,
                bboxes=target['boxes'],
                labels=labels,
            )
            if len(sample['bboxes']) > 0:         # at least one box survived
                image = sample['image']
                boxes = torch.stack(
                    tuple(map(torch.tensor, zip(*sample['bboxes'])))
                ).permute(1, 0)
                # xyxy -> yxyx for EfficientDet (skip if your model uses xyxy)
                boxes[:, [0, 1, 2, 3]] = boxes[:, [1, 0, 3, 2]]
                target['boxes'] = boxes
                break
        # else: fall through with the un-augmented image+target

    return image, target, labels
```

## Workflow

1. Wrap the entire `self.transforms(...)` call inside a `for _ in range(10)` loop
2. After each call, check `len(sample['bboxes']) > 0` — if true, accept and break
3. If the loop exhausts without a hit, return the un-augmented original (don't raise — empty targets crash training)
4. Convert albumentations' list-of-tuples bbox format to the tensor layout your model expects (xyxy, yxyx, cxcywh)
5. Apply only to the *training* dataset — validation should always pass through un-augmented

## Key Decisions

- **10 retries, not 100**: any aug pipeline that fails 10 times in a row is mis-tuned (e.g., crop too small for the smallest box); silently looping forever hides the bug.
- **Fall through, don't raise**: a missed augmentation on one sample is fine; crashing the whole epoch is not.
- **Belongs in `__getitem__`, not the collate_fn**: collate runs after batching, so you'd have to redo the entire batch — wasteful.
- **vs. `min_visibility` parameter**: `min_visibility=0.3` reduces but doesn't eliminate empty-box outcomes; the retry is the safety net.
- **Apply per-sample, not per-batch**: identical aug across a batch is much weaker regularization.

## References

- [2Class Object Detection Training](https://www.kaggle.com/competitions/nfl-impact-detection)
