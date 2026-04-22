---
name: nlp-learned-attention-pooling
description: Replace mean pooling with a trainable attention network (Linear-Tanh-Linear-Softmax) that learns token importance weights over transformer hidden states
---

# Learned Attention Pooling

## Overview

Mean pooling treats every token equally, which dilutes signal from key tokens. A lightweight trainable attention layer learns which tokens matter most for the task. The network computes a scalar weight per token via `Linear → Tanh → Linear → Softmax`, then produces a weighted sum of hidden states. Adds minimal parameters (~500K) but can significantly improve regression and classification tasks.

## Quick Start

```python
import torch.nn as nn

class AttentionPooling(nn.Module):
    def __init__(self, hidden_size, attn_size=512):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, attn_size),
            nn.Tanh(),
            nn.Linear(attn_size, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, hidden_states, attention_mask=None):
        # hidden_states: (batch, seq_len, hidden_size)
        weights = self.attention(hidden_states)  # (batch, seq_len, 1)
        if attention_mask is not None:
            weights = weights * attention_mask.unsqueeze(-1)
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        return (weights * hidden_states).sum(dim=1)  # (batch, hidden_size)

class Model(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        h = self.encoder.config.hidden_size
        self.pool = AttentionPooling(h)
        self.head = nn.Linear(h, 1)

    def forward(self, **inputs):
        hidden = self.encoder(**inputs).last_hidden_state
        pooled = self.pool(hidden, inputs.get('attention_mask'))
        return self.head(pooled)
```

## Workflow

1. Replace mean/CLS pooling with `AttentionPooling` module
2. Initialize attention weights with Xavier/He init
3. Apply attention mask to zero out padding tokens before softmax
4. Use the weighted sum as input to the task head

## Key Decisions

- **Attention hidden size**: 512 is standard; smaller (128) for small models, larger for XL models
- **Mask handling**: always mask padding tokens — without masking, attention leaks to pad positions
- **Init**: use `nn.init.xavier_uniform_` on linear layers for stable training
- **vs multi-head**: this is single-head; use `nlp-attention-head-pooling` for multi-head variant

## References

- [PPPM / Deberta-v3-large baseline w/ W&B [train]](https://www.kaggle.com/code/yasufuminakama/pppm-deberta-v3-large-baseline-w-w-b-train)
- [PPPM / Deberta-v3-large baseline [inference]](https://www.kaggle.com/code/yasufuminakama/pppm-deberta-v3-large-baseline-inference)
