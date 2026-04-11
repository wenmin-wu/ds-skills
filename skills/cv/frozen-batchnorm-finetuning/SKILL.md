---
name: cv-frozen-batchnorm-finetuning
description: >
  Unfreezes backbone layers for fine-tuning while keeping BatchNorm layers frozen to preserve pretrained running statistics.
---
# Frozen BatchNorm Fine-Tuning

## Overview

When fine-tuning a pretrained backbone on a small medical/domain-specific dataset, unfreezing all layers including BatchNorm can destabilize training. BN layers maintain running mean/variance from ImageNet — updating these with a small batch from a different domain collapses the statistics. The fix: unfreeze convolutional/linear layers for gradient updates but explicitly keep all BatchNorm layers frozen (eval mode). This preserves pretrained normalization while allowing the rest of the network to adapt.

## Quick Start

```python
import torch.nn as nn

def unfreeze_with_frozen_bn(model, n_layers=20):
    """Unfreeze last N backbone layers, keep all BatchNorm frozen."""
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last N layers, skipping BN
    layers = list(model.features.children())
    for layer in layers[-n_layers:]:
        for name, module in layer.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.eval()  # keep in eval mode
                for param in module.parameters():
                    param.requires_grad = False
            else:
                for param in module.parameters():
                    param.requires_grad = True

    # Always unfreeze the classifier head
    for param in model.classifier.parameters():
        param.requires_grad = True

# Also override train() to keep BN in eval mode
class FrozenBNModel(nn.Module):
    def train(self, mode=True):
        super().train(mode)
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.eval()
        return self
```

## Workflow

1. Load a pretrained model (EfficientNet, DenseNet, ResNet)
2. Freeze all parameters
3. Unfreeze the last N backbone layers — but skip BatchNorm modules
4. Always unfreeze the classifier head
5. Override `train()` to force BN layers into eval mode every forward pass

## Key Decisions

- **N layers**: Start with last 20; increase if underfitting, decrease if unstable
- **Override train()**: Critical — PyTorch's `model.train()` re-enables BN by default
- **GroupNorm/LayerNorm**: These don't use running statistics — safe to unfreeze
- **Small datasets**: More important when training data < 5000 images

## References

- [RSNA EfficientNet Starter Notebook](https://www.kaggle.com/code/shubhamcodez/rsna-efficientnet-starter-notebook)
