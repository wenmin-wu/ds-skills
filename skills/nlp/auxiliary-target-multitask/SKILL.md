---
name: nlp-auxiliary-target-multitask
description: Train main target alongside auxiliary sub-type targets as multiple output heads to regularize and improve generalization
domain: nlp
---

# Auxiliary Target Multi-Task Training

## Overview

When related labels exist (e.g., toxicity + severe_toxicity + obscene + insult), train them as auxiliary outputs sharing the same backbone. The auxiliary heads act as implicit regularization, forcing the shared representation to capture more general patterns. At inference, use only the main head.

## Quick Start

```python
import torch
import torch.nn as nn

class MultiTaskModel(nn.Module):
    def __init__(self, backbone, hidden_dim, n_aux_targets):
        super().__init__()
        self.backbone = backbone
        self.main_head = nn.Linear(hidden_dim, 1)
        self.aux_head = nn.Linear(hidden_dim, n_aux_targets)
    
    def forward(self, x):
        features = self.backbone(x)
        main_out = self.main_head(features)
        aux_out = self.aux_head(features)
        return main_out, aux_out

# Training
aux_columns = ['severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']
y_aux = train[aux_columns].values

for x_batch, y_main, y_aux_batch in dataloader:
    main_pred, aux_pred = model(x_batch)
    loss_main = nn.BCEWithLogitsLoss()(main_pred, y_main)
    loss_aux = nn.BCEWithLogitsLoss()(aux_pred, y_aux_batch)
    loss = loss_main + loss_aux  # equal weighting
    loss.backward()

# Inference: only use main head
main_pred, _ = model(x_test)
```

## Key Decisions

- **Equal loss weighting**: simple and effective; tune if aux targets are noisy
- **Shared backbone**: all heads share the encoder — this is what provides regularization
- **Discard aux at inference**: auxiliary heads are training-only scaffolding
- **Label availability**: works even if aux labels are partially missing — mask the loss

## References

- Source: [simple-lstm-pytorch-version](https://www.kaggle.com/code/bminixhofer/simple-lstm-pytorch-version)
- Competition: Jigsaw Unintended Bias in Toxicity Classification
