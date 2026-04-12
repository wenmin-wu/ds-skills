---
name: cv-bev-conv1-channel-expansion
description: Replace a pretrained ResNet's conv1 with a wider input conv to accept stacked BEV raster channels (semantic map + agent history) while keeping downstream weights
---

## Overview

Pretrained ImageNet backbones expect 3 RGB channels. Bird's-eye-view motion prediction rasters need many more: 3 for the semantic map plus `(history_frames + 1) * 2` channels for stacked ego + agent masks at past timesteps — typically 10-30 channels. Instead of retraining from scratch, swap only the first conv to the new input width. Downstream layers (conv2, blocks, etc.) retain ImageNet pretraining and still converge fast. The new conv1 is randomly initialized, so give it a slightly higher learning rate during warmup.

## Quick Start

```python
import torch.nn as nn
from torchvision.models import resnet34

class BEVBackbone(nn.Module):
    def __init__(self, history_num_frames, num_map_channels=3, pretrained=True):
        super().__init__()
        num_history_channels = (history_num_frames + 1) * 2
        num_in_channels = num_map_channels + num_history_channels

        self.backbone = resnet34(pretrained=pretrained)
        old = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_channels=num_in_channels,
            out_channels=old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=False,
        )
        # Optional: copy RGB weights onto the first 3 planes
        if pretrained:
            with torch.no_grad():
                self.backbone.conv1.weight[:, :3] = old.weight

    def forward(self, x):
        return self.backbone(x)
```

## Workflow

1. Compute `num_in_channels = map_channels + (history_frames + 1) * 2`
2. Instantiate the pretrained backbone, then replace only `conv1` with a new `Conv2d` matching the old kernel size/stride/padding but the new `in_channels`
3. (Optional) Copy the old 3-channel conv weights onto the first 3 input planes to preserve RGB pretraining
4. Fine-tune normally — the new conv1 learns in the first few thousand steps
5. Keep the rest of the backbone unchanged; its weights stay useful because they see the same feature maps downstream

## Key Decisions

- **Replace conv1 only**: replacing more layers throws away pretraining. One conv swap is enough.
- **Copy or random init**: copying the 3 RGB planes helps if the semantic map is rendered as RGB. Random init works fine if it's not.
- **No bias**: match the original ResNet which uses `bias=False` (BN follows).
- **vs. 1x1 adapter**: a 1x1 channel-reducer before conv1 preserves pretrained weights fully but adds a layer and is slower in practice.

## References

- [Pytorch Baseline - Train](https://www.kaggle.com/code/pestipeti/pytorch-baseline-train)
- [Lyft: Complete train and prediction pipeline](https://www.kaggle.com/code/pestipeti/lyft-complete-train-and-prediction-pipeline)
