---
name: cv-cnn-encoder-spatial-feature-map
description: >
  Strips the global pool and FC head from a pretrained CNN to expose spatial feature maps (H x W x C) for attention-based decoding.
---
# CNN Encoder Spatial Feature Map

## Overview

For image-to-sequence tasks (captioning, OCR, molecular translation), the CNN must output spatial feature maps rather than a single vector. Replace the global pooling and FC head with `nn.Identity()`, then permute/reshape the output to (batch, H*W, C). Each spatial position becomes an "input token" the decoder can attend to.

## Quick Start

```python
import timm
import torch.nn as nn

class SpatialEncoder(nn.Module):
    def __init__(self, model_name="resnet34", pretrained=True):
        super().__init__()
        self.cnn = timm.create_model(model_name, pretrained=pretrained)
        self.n_features = self.cnn.fc.in_features
        self.cnn.global_pool = nn.Identity()
        self.cnn.fc = nn.Identity()

    def forward(self, x):
        features = self.cnn(x)                      # (B, C, H, W)
        features = features.permute(0, 2, 3, 1)     # (B, H, W, C)
        B, H, W, C = features.shape
        features = features.view(B, H * W, C)       # (B, num_pixels, C)
        return features
```

## Workflow

1. Load any pretrained CNN via timm/torchvision
2. Replace `global_pool` and `fc` with `nn.Identity()`
3. Forward pass yields (B, C, H, W) spatial tensor
4. Permute and reshape to (B, num_pixels, C) for attention input
5. Feed spatial features into attention-based decoder

## Key Decisions

- **Model choice**: ResNet-34/50 gives 7x7=49 spatial positions for 224x224 input; EfficientNet varies
- **Image size**: Larger input = more spatial positions = richer attention but more memory
- **Adaptive pooling**: Optionally add `nn.AdaptiveAvgPool2d((H, W))` before flatten for fixed spatial dims
- **Fine-tuning**: Freeze early layers, train later layers with the decoder

## References

- [InChI / Resnet + LSTM with attention / starter](https://www.kaggle.com/code/yasufuminakama/inchi-resnet-lstm-with-attention-starter)
- [Pytorch ResNet+LSTM with attention](https://www.kaggle.com/code/pasewark/pytorch-resnet-lstm-with-attention)
