---
name: nlp-transformer-layer-freezing
description: >
  Freezes transformer embedding and lower encoder layers to reduce memory, speed up training, and stabilize fine-tuning.
---
# Transformer Layer Freezing

## Overview

Lower transformer layers learn general language features that transfer well; upper layers are task-specific. Freeze embeddings and the first N encoder layers to skip gradient computation on stable parameters. This reduces GPU memory, speeds up training, and prevents catastrophic forgetting — especially useful when fine-tuning large models on small datasets.

## Quick Start

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("microsoft/deberta-v3-large")

def freeze(module):
    for param in module.parameters():
        param.requires_grad = False

# Freeze embeddings + first 2 encoder layers
freeze(model.embeddings)
freeze(model.encoder.layer[:2])

# Only pass trainable params to optimizer
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable_params, lr=2e-5)
```

## Workflow

1. Load pretrained transformer model
2. Freeze `model.embeddings` (word, position, token_type)
3. Freeze first N encoder layers via `model.encoder.layer[:N]`
4. Filter `requires_grad=True` parameters for optimizer
5. Train normally — frozen layers use zero memory for gradients/optimizer states

## Key Decisions

- **Layers to freeze**: 2-4 for base models, 4-8 for large; more freezing = faster but less adaptation
- **Gradual unfreezing**: Start frozen, unfreeze one layer per epoch for smoother convergence
- **Combine with layerwise LR decay**: Freeze lowest layers, apply decreasing LR to middle layers
- **Memory savings**: ~30-50% reduction in optimizer states per frozen layer

## References

- [Optimization approaches for Transformers](https://www.kaggle.com/code/vad13irt/optimization-approaches-for-transformers)
- [Huge Ensemble](https://www.kaggle.com/code/thedevastator/huge-ensemble)
