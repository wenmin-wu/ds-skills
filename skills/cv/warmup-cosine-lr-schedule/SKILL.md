---
name: cv-warmup-cosine-lr-schedule
description: >
  Combines gradual learning rate warmup with cosine annealing decay for stable fine-tuning of pretrained models.
---
# Warmup + Cosine Annealing LR Schedule

## Overview

Start training with a low learning rate, ramp up linearly over a warmup period, then decay following a cosine curve. The warmup prevents large gradient updates from destroying pretrained weights in early steps. Cosine decay provides smooth, aggressive LR reduction that works better than step decay for fine-tuning.

## Quick Start

```python
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

def get_cosine_with_warmup(optimizer, warmup_epochs, total_epochs, min_lr=1e-7):
    """Cosine annealing with linear warmup."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return max(min_lr / optimizer.defaults['lr'],
                   0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item()))
    return LambdaLR(optimizer, lr_lambda)

# Usage
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
scheduler = get_cosine_with_warmup(optimizer, warmup_epochs=2, total_epochs=30)

for epoch in range(30):
    train_one_epoch(model, train_loader, optimizer)
    scheduler.step()
```

## Workflow

1. Set peak learning rate (e.g., 3e-4 for Adam, 1e-2 for SGD)
2. Warmup linearly from ~0 to peak over 1-3 epochs
3. Cosine decay from peak to min_lr over remaining epochs
4. Step scheduler once per epoch (not per batch)

## Key Decisions

- **Warmup duration**: 1-3 epochs for fine-tuning; 5-10% of total steps for training from scratch
- **Min LR**: 1e-7 is safe; too high (1e-5) and the model keeps updating when it should converge
- **vs OneCycleLR**: OneCycle has aggressive warmup + cooldown; cosine+warmup is more conservative
- **Per-batch warmup**: For very short training, warmup per step instead of per epoch

## References

- RANZCR CLiP - Catheter and Line Position Challenge (Kaggle)
- Source: [single-fold-training-of-resnet200d-lb0-965](https://www.kaggle.com/code/underwearfitting/single-fold-training-of-resnet200d-lb0-965)
