---
name: cv-efficientdet-headnet-swap
description: Load EfficientDet pretrained on COCO with the original 90-class head, then swap in a fresh HeadNet with your own num_classes — keeps the BiFPN feature pyramid pretrained and only retrains the classification head, the canonical transfer-learning recipe for the effdet PyTorch port
---

## Overview

The default `effdet` PyTorch package loads EfficientDet checkpoints assuming COCO's 90 classes. If you instantiate with `num_classes=2` directly, the state dict load fails because the head shapes don't match. The right pattern is the reverse: instantiate with the original config, load the COCO weights, *then* mutate `config.num_classes` and replace `net.class_net` with a fresh `HeadNet`. The BiFPN and backbone keep their pretrained weights, the classification head re-initializes for your task, and `DetBenchTrain` wraps everything in the loss-computing forward pass. This is the canonical 4-line recipe for fine-tuning EfficientDet on any custom detection dataset.

## Quick Start

```python
import torch
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet

def get_net(num_classes=2, image_size=512, ckpt='efficientdet_d5-ef44aea8.pth'):
    config = get_efficientdet_config('tf_efficientdet_d5')
    net = EfficientDet(config, pretrained_backbone=False)

    checkpoint = torch.load(ckpt, map_location='cpu')
    net.load_state_dict(checkpoint)               # load with original 90 classes

    config.num_classes = num_classes
    config.image_size  = image_size
    net.class_net = HeadNet(
        config,
        num_outputs=config.num_classes,
        norm_kwargs=dict(eps=1e-3, momentum=0.01),
    )
    return DetBenchTrain(net, config)
```

## Workflow

1. Pick the EfficientDet variant that matches your compute — D0 for prototyping, D5 for production accuracy
2. `get_efficientdet_config('tf_efficientdet_d5')` and instantiate the model with the *original* COCO classes
3. `load_state_dict(checkpoint)` — must happen before mutating `num_classes`, otherwise shapes mismatch
4. Mutate `config.num_classes` and `config.image_size` to your task's values
5. Replace `net.class_net` with a freshly-initialized `HeadNet` of the new shape
6. Wrap in `DetBenchTrain` for training (adds the loss heads) or `DetBenchPredict` for inference
7. Train with a low LR on the new head and a 10x lower LR on the rest (use param groups)

## Key Decisions

- **Load COCO weights first, swap head second**: doing it the other way fails the state-dict shape check. Counterintuitive but correct.
- **`pretrained_backbone=False`**: the backbone weights are inside the COCO checkpoint already; setting True double-loads and slows init.
- **`norm_kwargs=dict(eps=1e-3, momentum=0.01)`**: matches the BatchNorm config the original COCO weights expect; default torch BN values cause silent train/eval mismatch.
- **Don't replace `box_net`**: only the classification head depends on `num_classes`; the regression head is class-agnostic.
- **Train at the same image size as `config.image_size`**: EfficientDet anchors are precomputed for the configured size; mismatched sizes produce empty positives.

## References

- [2Class Object Detection Training](https://www.kaggle.com/competitions/nfl-impact-detection)
