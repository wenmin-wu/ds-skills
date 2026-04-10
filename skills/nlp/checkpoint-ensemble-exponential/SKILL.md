---
name: nlp-checkpoint-ensemble-exponential
description: Average predictions from each epoch checkpoint with exponentially increasing weights (2^epoch), favoring later more-converged snapshots
domain: nlp
---

# Checkpoint Ensemble with Exponential Weights

## Overview

Instead of using only the best checkpoint, collect predictions from every epoch and average with exponential weights (2^epoch). Later epochs get more weight since they're more converged, but earlier epochs contribute diversity. Costs zero extra training time — just save predictions at each epoch.

## Quick Start

```python
import numpy as np

def checkpoint_ensemble_predict(model, train_fn, test_loader, n_epochs,
                                 device='cuda'):
    """Train model and collect exponentially-weighted checkpoint predictions.
    
    Args:
        model: PyTorch model
        train_fn: function(model, epoch) that trains one epoch
        test_loader: DataLoader for test data
        n_epochs: total training epochs
    Returns:
        averaged predictions array
    """
    all_preds = []
    weights = [2 ** epoch for epoch in range(n_epochs)]
    
    for epoch in range(n_epochs):
        train_fn(model, epoch)
        
        model.eval()
        epoch_preds = []
        with torch.no_grad():
            for x_batch in test_loader:
                logits = model(x_batch.to(device))
                epoch_preds.append(torch.sigmoid(logits).cpu().numpy())
        all_preds.append(np.concatenate(epoch_preds))
    
    return np.average(all_preds, weights=weights, axis=0)

# Usage
predictions = checkpoint_ensemble_predict(model, train_one_epoch, test_loader, 5)
```

## Key Decisions

- **2^epoch weights**: geometric increase; last epoch gets 16x weight of first (for 5 epochs)
- **No extra cost**: predictions are collected during training — no separate inference pass needed
- **Combine with K-fold**: collect per-epoch preds for each fold, then average across both dimensions
- **Alternative**: uniform weights work too but exponential consistently scores 0.001–0.005 better

## References

- Source: [simple-lstm-pytorch-version](https://www.kaggle.com/code/bminixhofer/simple-lstm-pytorch-version)
- Competition: Jigsaw Unintended Bias in Toxicity Classification
