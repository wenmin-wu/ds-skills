---
name: nlp-lstm-over-transformer-layers
description: >
  Feeds CLS token embeddings from each transformer layer into a BiLSTM to learn an optimal combination across layer depth.
---
# LSTM Over Transformer Layers

## Overview

Different transformer layers capture different levels of abstraction. Instead of using only the last layer or a weighted sum, stack CLS token embeddings from all layers into a sequence (layer 1 → layer N) and pass through a BiLSTM. The LSTM learns which layers matter and how to combine them — often outperforming static pooling strategies.

## Quick Start

```python
import torch.nn as nn

class LSTMPooling(nn.Module):
    def __init__(self, num_layers, hidden_size, lstm_hidden=256):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(hidden_size, lstm_hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.1)

    def forward(self, all_hidden_states):
        # Stack CLS token from each layer: (batch, num_layers, hidden_size)
        cls_per_layer = torch.stack(
            [all_hidden_states[i][:, 0] for i in range(1, self.num_layers + 1)], dim=1
        )
        out, _ = self.lstm(cls_per_layer)
        return self.dropout(out[:, -1, :])  # last LSTM output

# Usage in model forward:
outputs = transformer(input_ids, attention_mask, output_hidden_states=True)
pooled = lstm_pooling(outputs.hidden_states)
logits = classifier_head(pooled)
```

## Workflow

1. Enable `output_hidden_states=True` in the transformer forward pass
2. Extract CLS token (position 0) from each hidden state layer
3. Stack into (batch, num_layers, hidden_size) tensor
4. Pass through BiLSTM; take final output as the pooled representation
5. Feed into classification/regression head

## Key Decisions

- **Bidirectional**: BiLSTM captures both shallow→deep and deep→shallow patterns
- **LSTM hidden dim**: 256 is typical; output is 512 for bidirectional (2x)
- **vs weighted-layer-pooling**: LSTM is more expressive but adds trainable parameters
- **Layer selection**: Can use all layers or skip every other for efficiency

## References

- [feedback_deberta_large_LB0.619](https://www.kaggle.com/code/brandonhu0215/feedback-deberta-large-lb0-619)
