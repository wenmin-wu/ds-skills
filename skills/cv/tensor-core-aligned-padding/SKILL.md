---
name: cv-tensor-core-aligned-padding
description: Pad batch sequence lengths to multiples of 8 for efficient tensor core utilization on GPUs, with -100 masking for label padding
---

# Tensor Core Aligned Padding

## Overview

NVIDIA tensor cores operate most efficiently on dimensions that are multiples of 8 (FP16) or 16 (INT8). When creating batches with variable-length sequences, pad to the next multiple of 8 instead of just the max length. Combined with masking padding tokens as -100 in labels (ignored by CrossEntropyLoss), this yields up to 15% throughput improvement with no accuracy impact.

## Quick Start

```python
import torch

def collate_aligned(samples, pad_token_id, align=8):
    max_len = max(len(s["input_ids"]) for s in samples)
    if max_len % align != 0:
        max_len = (max_len // align + 1) * align

    input_ids = []
    for s in samples:
        padded = s["input_ids"] + [pad_token_id] * (max_len - len(s["input_ids"]))
        input_ids.append(padded)

    input_ids = torch.tensor(input_ids)
    labels = input_ids.clone()
    labels[labels == pad_token_id] = -100  # ignore padding in loss

    attention_mask = (input_ids != pad_token_id).long()
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
```

## Workflow

1. Find max sequence length in the batch
2. Round up to next multiple of 8
3. Pad all sequences to the aligned length with pad_token_id
4. Clone padded input_ids as labels, replace pad positions with -100
5. Create attention mask from non-pad positions

## Key Decisions

- **Align=8**: optimal for FP16 on NVIDIA Ampere/Hopper; use 16 for INT8 quantized models
- **-100 masking**: PyTorch CrossEntropyLoss ignores index -100 by default
- **Dynamic padding**: still pad to batch max (not global max) — align just rounds up slightly
- **When it matters**: most impactful for long sequences (512+) and large batch sizes

## References

- [donut-train [benetech]](https://www.kaggle.com/code/nbroad/donut-train-benetech)
