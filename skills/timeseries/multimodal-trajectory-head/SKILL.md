---
name: timeseries-multimodal-trajectory-head
description: Single linear head that jointly predicts K candidate trajectories and K softmax confidences, sliced and reshaped for multimodal regression
---

## Overview

Rather than using K separate prediction heads (one per mode) or a complex MoE router, pack all K trajectory predictions plus K confidence logits into a single linear head with `K*T*2 + K` outputs. Slice the output tensor into two chunks, reshape the first into `(B, K, T, 2)`, and softmax the second into mode probabilities. This keeps the model architecture simple, parameters shared across modes, and adds one `Linear` layer to any CNN/Transformer backbone. Pairs directly with the K-mode Gaussian NLL loss.

## Quick Start

```python
import torch
import torch.nn as nn
from torchvision.models import resnet34

class MultiModalTrajModel(nn.Module):
    def __init__(self, backbone_features=512, num_modes=3, future_len=50):
        super().__init__()
        self.num_modes = num_modes
        self.future_len = future_len
        self.num_preds = num_modes * 2 * future_len
        self.backbone = resnet34(pretrained=True)
        self.backbone.fc = nn.Identity()
        self.head = nn.Linear(backbone_features, self.num_preds + num_modes)

    def forward(self, x):
        f = self.backbone(x)
        out = self.head(f)                              # (B, K*T*2 + K)
        pred, conf = out[:, :self.num_preds], out[:, self.num_preds:]
        pred = pred.view(-1, self.num_modes, self.future_len, 2)
        conf = torch.softmax(conf, dim=1)
        return pred, conf
```

## Workflow

1. Set the output dim of the final linear layer to `num_modes * 2 * future_len + num_modes`
2. Split the output on the feature axis at `num_modes * 2 * future_len`
3. Reshape the first chunk to `(B, K, T, 2)` — the K candidate trajectories
4. Apply softmax along `dim=1` to the second chunk — the mode mixture weights
5. Feed `(pred, conf)` into the K-mode NLL loss

## Key Decisions

- **Shared backbone, single head**: K modes share all features; the head only has `~K*T*2` extra weights vs. a single-mode model.
- **Softmax, not sigmoid**: confidences must sum to 1 to form a valid mixture.
- **Reshape via `.view()`**: requires contiguous tensors. Use `.reshape()` if upstream ops produce non-contiguous output.
- **vs. K independent heads**: K heads produces K independent feature transforms that often collapse to the same mean mode. Shared features + slice-and-reshape forces diversity via the NLL loss instead.

## References

- [Pytorch Baseline - Train](https://www.kaggle.com/code/pestipeti/pytorch-baseline-train)
- [Lyft: Complete train and prediction pipeline](https://www.kaggle.com/code/pestipeti/lyft-complete-train-and-prediction-pipeline)
