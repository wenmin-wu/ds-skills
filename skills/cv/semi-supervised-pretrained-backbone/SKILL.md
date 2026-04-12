---
name: cv-semi-supervised-pretrained-backbone
description: >
  Uses Facebook's semi-weakly supervised ImageNet-pretrained models (trained on 940M unlabeled images) as CNN backbones for stronger transfer learning than standard supervised pretraining.
---
# Semi-Supervised Pretrained Backbone

## Overview

Facebook's semi-supervised and semi-weakly supervised ImageNet models were trained on up to 940M unlabeled images from YFCC100M, producing significantly better representations than standard supervised pretraining. Using these as backbones (ResNet-50, ResNeXt-50) provides 1-3% accuracy improvements on downstream tasks, especially on medical imaging and domain-shifted data where standard ImageNet features are weak. Available via `torch.hub` with no extra dependencies.

## Quick Start

```python
import torch
import torch.nn as nn

# Load semi-weakly supervised ResNeXt-50
backbone = torch.hub.load(
    'facebookresearch/semi-supervised-ImageNet1K-models',
    'resnext50_32x4d_swsl'  # swsl = semi-weakly supervised
)

class CustomModel(nn.Module):
    def __init__(self, n_classes=6):
        super().__init__()
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])
        nc = list(backbone.children())[-1].in_features  # 2048
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(nc, 512), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(512, n_classes))

    def forward(self, x):
        return self.head(self.encoder(x))

model = CustomModel(n_classes=6)
```

## Workflow

1. Load pretrained model via `torch.hub.load`
2. Strip the final classification layer
3. Add custom pooling + classification head
4. Fine-tune end-to-end on target dataset

## Key Decisions

- **Model variants**: `resnext50_32x4d_ssl` (semi-supervised), `resnext50_32x4d_swsl` (semi-weakly supervised — stronger)
- **vs supervised**: SWSL models consistently outperform supervised on transfer tasks
- **vs DINO/MAE**: SWSL is older but simpler to use; no special fine-tuning needed
- **Available architectures**: ResNet-18/50, ResNeXt-50/101 in both SSL and SWSL variants

## References

- [PANDA concat tile pooling starter](https://www.kaggle.com/code/iafoss/panda-concat-tile-pooling-starter-0-79-lb)
