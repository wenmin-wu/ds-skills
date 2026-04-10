---
name: nlp-last-hidden-states-concat
description: Concatenate the last two transformer hidden states along the feature dimension before the task head for richer token representations
domain: nlp
---

# Last Hidden States Concatenation

## Overview

Instead of using only the final hidden state from a transformer encoder, concatenate the last two (or more) hidden layers along the feature dimension. The second-to-last layer often captures different linguistic patterns than the final layer. This doubles the representation size but consistently improves span extraction and token classification tasks with minimal overhead.

## Quick Start

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class SpanExtractor(nn.Module):
    def __init__(self, model_name, n_layers=2):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        self.encoder = AutoModel.from_pretrained(model_name, config=config)
        hidden_size = config.hidden_size * n_layers
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(hidden_size, 2)  # start + end logits
        nn.init.normal_(self.head.weight, std=0.02)
        self.n_layers = n_layers

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.encoder(
            input_ids, attention_mask=attention_mask,
            token_type_ids=token_type_ids)
        hidden_states = outputs.hidden_states
        # Concatenate last N layers
        cat = torch.cat(hidden_states[-self.n_layers:], dim=-1)
        logits = self.head(self.dropout(cat))
        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)
```

## Key Decisions

- **2 layers is standard**: last 2 is the sweet spot; 3-4 adds parameters with diminishing returns
- **output_hidden_states=True**: must enable in config to access intermediate layers
- **Linear head doubles**: input size to head is `hidden_size * n_layers`
- **Alternative: weighted sum**: use `nlp-weighted-layer-pooling` for learned weights instead of concat

## References

- Source: [roberta-inference-5-folds](https://www.kaggle.com/code/abhishek/roberta-inference-5-folds)
- Competition: Tweet Sentiment Extraction
