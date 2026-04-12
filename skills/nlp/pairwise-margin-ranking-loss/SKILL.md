---
name: nlp-pairwise-margin-ranking-loss
description: >
  Trains a transformer with MarginRankingLoss on text pairs (more/less toxic), learning to rank rather than classify when only pairwise preference labels are available.
---
# Pairwise Margin Ranking Loss

## Overview

When labels are pairwise preferences ("text A is more toxic than text B") rather than absolute scores, MarginRankingLoss trains a model to produce scalar scores where the preferred item scores higher by at least a margin. Each text is encoded independently through a shared transformer, producing two scalars per pair. The loss penalizes pairs where the "more toxic" score isn't at least `margin` above the "less toxic" score. This is the standard approach for learning-to-rank with neural encoders.

## Quick Start

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class RankingModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, ids, mask):
        out = self.encoder(input_ids=ids, attention_mask=mask)
        return self.fc(self.drop(out.pooler_output))

# Training
criterion = nn.MarginRankingLoss(margin=0.5)
target = torch.ones(batch_size)  # more_toxic should score higher

score_more = model(more_toxic_ids, more_toxic_mask)
score_less = model(less_toxic_ids, less_toxic_mask)
loss = criterion(score_more.squeeze(), score_less.squeeze(), target)
```

## Workflow

1. Tokenize both texts in each pair independently
2. Forward each through the shared encoder to get scalar scores
3. Compute MarginRankingLoss(score_more, score_less, target=1)
4. At inference, rank all texts by their scalar score

## Key Decisions

- **Margin**: 0.3-1.0; larger margin forces stronger separation but may underfit
- **Shared encoder**: Both items use the same weights — this is a siamese architecture
- **Pooling**: CLS token or mean pooling both work; CLS is simpler for scalar output
- **vs classification**: Ranking loss doesn't need absolute labels, only relative ordering

## References

- [Pytorch + W&B Jigsaw Starter](https://www.kaggle.com/code/debarshichanda/pytorch-w-b-jigsaw-starter)
