---
name: nlp-weight-decay-bias-exclusion
description: Exclude bias terms and LayerNorm weights from weight decay to prevent regularization from distorting normalization layers
domain: nlp
---

# Weight Decay Bias Exclusion

## Overview

Weight decay (L2 regularization) should only apply to weight matrices, not to bias terms or LayerNorm parameters. Regularizing biases pushes them toward zero unnecessarily, and regularizing LayerNorm weights distorts the learned scale. Standard practice for fine-tuning transformers — split parameters into two groups with different decay rates.

## Quick Start

```python
from torch.optim import AdamW

def get_optimizer_grouped_parameters(model, lr=2e-5, weight_decay=0.01):
    """Split parameters into decay and no-decay groups."""
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    
    return [
        {
            'params': [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            'weight_decay': weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]

optimizer = AdamW(
    get_optimizer_grouped_parameters(model, lr=2e-5, weight_decay=0.01)
)
```

## Key Decisions

- **weight_decay=0.01**: standard for BERT/transformer fine-tuning; increase for stronger regularization
- **LayerNorm.weight included**: this is a scale parameter, not a typical weight — decaying it hurts normalization
- **String matching**: `any(nd in n ...)` checks parameter names — verify naming convention matches your model
- **Also applies to embeddings**: some practitioners exclude embedding weights too — experiment

## References

- Source: [toxic-bert-plain-vanila](https://www.kaggle.com/code/yuval6967/toxic-bert-plain-vanila)
- Competition: Jigsaw Unintended Bias in Toxicity Classification
