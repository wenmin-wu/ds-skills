---
name: cv-2d-to-3d-model-conversion
description: >
  Converts a pretrained 2D CNN into a 3D CNN by recursively replacing Conv2d/BN2d/Pool2d layers and inflating kernel weights along the depth axis.
---
# 2D to 3D Model Conversion

## Overview

Pretrained 3D CNNs are scarce, but pretrained 2D models (ImageNet) are abundant. This technique recursively walks a 2D model, replaces Conv2d with Conv3d, BatchNorm2d with BatchNorm3d, and pooling layers with their 3D counterparts. Crucially, 2D kernel weights are inflated by repeating along the new depth dimension, preserving learned spatial features. This gives 3D volumetric models (CT, MRI) a strong initialization without training from scratch.

## Quick Start

```python
import torch
import torch.nn as nn

def convert_3d(module):
    module_output = module
    if isinstance(module, nn.BatchNorm2d):
        module_output = nn.BatchNorm3d(module.num_features, module.eps,
                                        module.momentum, module.affine,
                                        module.track_running_stats)
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
            module_output.running_mean = module.running_mean
            module_output.running_var = module.running_var
    elif isinstance(module, nn.Conv2d):
        module_output = nn.Conv3d(
            module.in_channels, module.out_channels,
            kernel_size=module.kernel_size[0], stride=module.stride[0],
            padding=module.padding[0], dilation=module.dilation[0],
            groups=module.groups, bias=module.bias is not None)
        # Inflate: repeat 2D weights along depth
        module_output.weight = nn.Parameter(
            module.weight.unsqueeze(-1).repeat(1, 1, 1, 1, module.kernel_size[0]))
    elif isinstance(module, nn.MaxPool2d):
        module_output = nn.MaxPool3d(
            kernel_size=module.kernel_size, stride=module.stride,
            padding=module.padding)
    for name, child in module.named_children():
        module_output.add_module(name, convert_3d(child))
    return module_output

# Usage: inflate a pretrained EfficientNet to 3D
model_2d = torchvision.models.efficientnet_v2_s(weights="IMAGENET1K_V1")
model_3d = convert_3d(model_2d)
```

## Workflow

1. Load a pretrained 2D model (ResNet, EfficientNet, etc.)
2. Call `convert_3d(model)` to recursively replace all 2D layers with 3D equivalents
3. Inflate Conv2d weights by repeating along the kernel depth dimension
4. Attach a new classification/segmentation head for your 3D task
5. Fine-tune on volumetric data (CT scans, MRI volumes)

## Key Decisions

- **Weight inflation**: Repeating weights along depth preserves spatial features; averaging is an alternative
- **Input channels**: First conv may need adjustment if input has 1 channel (grayscale CT) vs 3 (RGB)
- **Pooling**: AdaptiveAvgPool2d must also be converted to AdaptiveAvgPool3d
- **When to use**: Medical imaging (CT, MRI), video classification, any 3D volumetric task

## References

- [RSNA 2022 1st Place Solution - Train Stage1](https://www.kaggle.com/code/haqishen/rsna-2022-1st-place-solution-train-stage1)
