---
name: timeseries-availability-masked-regression-loss
description: Multiply per-timestep regression loss by a 0/1 availability mask so missing future steps contribute zero gradient
---

## Overview

In trajectory forecasting, many ground-truth futures are incomplete: an agent exits the scene partway through the prediction horizon, or a sensor misses frames. Training with naive MSE over a fixed-length future either requires padding with bogus values (biases the model) or dropping whole samples (wastes data). The fix is a per-timestep availability mask that's 1 where GT exists and 0 where it doesn't. Use `reduction='none'`, multiply by the mask, and take the mean. Gradients flow only through valid steps, so the loss is unbiased without truncating the horizon.

## Quick Start

```python
import torch
import torch.nn as nn

criterion = nn.MSELoss(reduction='none')

def masked_loss(pred, target, avails):
    """
    pred:    (B, T, 2)
    target:  (B, T, 2)
    avails:  (B, T)   1/0 validity mask
    """
    loss = criterion(pred, target)             # (B, T, 2)
    loss = loss * avails.unsqueeze(-1)         # zero-out invalid steps
    return loss.mean()

# Training step
pred = model(batch['image']).view(batch['target_positions'].shape)
loss = masked_loss(pred,
                   batch['target_positions'],
                   batch['target_availabilities'])
loss.backward()
```

## Workflow

1. Ensure the dataloader yields a `target_availabilities` tensor alongside each target (shape `(B, T)`)
2. Use `reduction='none'` on the underlying loss so element-wise values survive
3. Broadcast the mask onto the final dim with `.unsqueeze(-1)`
4. Multiply — zero-weight invalid steps
5. Take the mean over all elements (or sum and divide by `avails.sum()` for a length-normalized version)

## Key Decisions

- **mean vs. sum/count**: `.mean()` divides by the total element count including masked zeros, so the effective learning rate scales with sequence length. For length-invariant loss, use `loss.sum() / (avails.sum() * D)`.
- **Broadcast the mask**: `(B, T, 1)` broadcasts cleanly to `(B, T, 2)` or any feature dim.
- **vs. ignore_index**: only works for classification; regression needs explicit masking.
- **vs. per-sample drop**: masking keeps partial trajectories in the training set, which is often >20% of data.

## References

- [Pytorch Baseline - Train](https://www.kaggle.com/code/pestipeti/pytorch-baseline-train)
