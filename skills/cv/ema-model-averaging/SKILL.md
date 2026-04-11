---
name: cv-ema-model-averaging
description: >
  Tracks an Exponential Moving Average of model weights during training and evaluates both live and EMA models for more stable predictions.
---
# EMA Model Averaging

## Overview

Exponential Moving Average (EMA) maintains a shadow copy of model weights as a running average: `ema_w = decay * ema_w + (1 - decay) * w`. The EMA model is smoother than the live model — it averages over many training steps, reducing variance from batch noise and learning rate spikes. At inference, the EMA weights typically outperform any single checkpoint by 0.002–0.01. During training, evaluate both live and EMA models to track both trajectories.

## Quick Start

```python
import torch
from copy import deepcopy

class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema = deepcopy(model)
        self.ema.eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ema_p, model_p in zip(self.ema.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)

    def state_dict(self):
        return self.ema.state_dict()

# Training loop
model = MyModel().cuda()
ema = ModelEMA(model, decay=0.999)

for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        loss = criterion(model(batch['image']), batch['target'])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        ema.update(model)  # update EMA after each step

    # Evaluate both
    val_score = evaluate(model, val_loader)
    ema_score = evaluate(ema.ema, val_loader)
    print(f"Live: {val_score:.4f}, EMA: {ema_score:.4f}")

# Save EMA weights for inference
torch.save(ema.state_dict(), 'model_ema.pth')
```

## Workflow

1. Initialize EMA as a deep copy of the model with `requires_grad=False`
2. After each optimizer step, call `ema.update(model)`
3. Evaluate both live and EMA models on validation set
4. Save EMA weights for inference (usually better than last checkpoint)

## Key Decisions

- **Decay**: 0.999 is standard; 0.9999 for long training; 0.99 for short/aggressive
- **Update frequency**: Every step is best; every N steps saves compute but reduces smoothing
- **BN buffers**: Also EMA-average batch norm running stats for consistency
- **Warmup**: Some implementations ramp decay from 0.99 to 0.999 over first 1000 steps

## References

- [VinBigData 2-Class Classifier Complete Pipeline](https://www.kaggle.com/code/corochann/vinbigdata-2-class-classifier-complete-pipeline)
