---
name: nlp-joint-span-answer-type-head
description: >
  Dual-head transformer model that jointly predicts start/end span logits from sequence output and answer type from pooled CLS output.
---
# Joint Span + Answer Type Head

## Overview

Open-domain QA requires both extracting an answer span and classifying the answer type (short answer, long answer, yes/no, or unanswerable). A joint model shares the transformer backbone and adds two heads: (1) a token-level Dense(2) split into start/end logits for span extraction, and (2) a Dense(N) on the CLS pooled output for answer type classification. Joint training improves both tasks through shared representations.

## Quick Start

```python
import torch
import torch.nn as nn
from transformers import AutoModel

class JointQAModel(nn.Module):
    def __init__(self, model_name, n_answer_types=5):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size
        self.span_head = nn.Linear(hidden, 2)       # start + end logits
        self.type_head = nn.Linear(hidden, n_answer_types)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (B, seq_len, H)
        pooled_output = outputs.last_hidden_state[:, 0]  # CLS token

        span_logits = self.span_head(sequence_output)  # (B, seq_len, 2)
        start_logits, end_logits = span_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)  # (B, seq_len)
        end_logits = end_logits.squeeze(-1)

        type_logits = self.type_head(pooled_output)  # (B, n_types)
        return start_logits, end_logits, type_logits
```

## Workflow

1. Pass input through shared transformer backbone
2. Apply Dense(2) to sequence output, split into start/end logits
3. Apply Dense(N) to CLS pooled output for answer type
4. Train with combined loss: span CE + type CE (weighted)
5. At inference, use type prediction to decide whether to emit a span

## Key Decisions

- **Answer types**: Typically 5 classes — UNKNOWN, YES, NO, SHORT, LONG
- **Loss weighting**: Equal weights work; tune if one task dominates
- **Pooled output**: Use CLS token or mean pooling for type head
- **Span constraints**: Enforce start < end and max span length at decode time

## References

- [BERT Joint Baseline Notebook](https://www.kaggle.com/code/prokaj/bert-joint-baseline-notebook)
- [TensorFlow 2.0 - Bert YES/NO Answers](https://www.kaggle.com/code/mmmarchetti/tensorflow-2-0-bert-yes-no-answers)
