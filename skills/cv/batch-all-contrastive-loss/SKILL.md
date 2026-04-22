---
name: cv-batch-all-contrastive-loss
description: All-vs-all contrastive loss comparing every pair in a batch (N^2 pairs) with margin and compactification regularizer
---

# Batch-All Contrastive Loss

## Overview

Instead of sampling specific positive/negative pairs, compute distances for all N^2 pairs in a batch. Positive pairs (same class) minimize distance; negative pairs push apart beyond margin m. A compactification term prevents embedding space from expanding unboundedly. Averaging only over non-zero loss terms focuses learning on informative pairs.

## Quick Start

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BatchAllContrastiveLoss(nn.Module):
    def __init__(self, margin=10.0, wd=1e-4):
        super().__init__()
        self.margin = margin
        self.wd = wd

    def forward(self, embeddings, labels):
        n = embeddings.size(0)
        dist = torch.cdist(embeddings, embeddings).pow(2).view(-1)
        labels_eq = (labels.unsqueeze(0) == labels.unsqueeze(1))
        eye = torch.eye(n, device=labels.device).bool()
        pos_mask = (labels_eq | eye).view(-1)

        loss_pos = dist[pos_mask]
        loss_neg = F.relu(self.margin - dist[~pos_mask].sqrt()).pow(2)
        all_loss = torch.cat([loss_pos, loss_neg])
        nonzero = all_loss[all_loss > 0]
        loss = nonzero.mean() if nonzero.numel() > 0 else all_loss.sum()
        loss += self.wd * dist.mean()
        return loss
```

## Workflow

1. Forward batch through embedding model
2. Compute all N^2 pairwise squared distances
3. Split into positive pairs (same label) and negative pairs (different label)
4. Positive loss = squared distance; negative loss = relu(margin - distance)^2
5. Average only non-zero terms + compactification regularizer

## Key Decisions

- **Margin**: 10.0 is a common starting point; tune based on embedding dimensionality
- **Non-zero averaging**: ignores already-satisfied constraints, focusing gradients on hard cases
- **Compactification**: `wd * mean(dist^2)` prevents embeddings from drifting to infinity
- **N^2 scaling**: effective for batch sizes up to ~256; beyond that, sample pairs

## References

- [Similarity DenseNet121 [0.805LB]](https://www.kaggle.com/code/iafoss/similarity-densenet121-0-805lb-kernel-time-limit)
