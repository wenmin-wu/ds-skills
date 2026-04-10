---
name: nlp-8bit-optimizer-embedding-override
description: >
  Uses bitsandbytes 8-bit AdamW to halve optimizer memory, with a 32-bit override for embedding weights to prevent instability.
---
# 8-Bit Optimizer with Embedding Override

## Overview

Optimizer states (momentum + variance in Adam) consume 2x the model size. BitsAndBytes 8-bit AdamW quantizes these states to 8-bit, cutting optimizer memory by ~75%. However, embedding layers are sensitive to quantization noise. Override embedding weights to use 32-bit optimizer states while keeping everything else at 8-bit.

## Quick Start

```python
import bitsandbytes as bnb

def set_embedding_parameters_bits(embeddings, optim_bits=32):
    for attr in ("word_embeddings", "position_embeddings", "token_type_embeddings"):
        if hasattr(embeddings, attr):
            bnb.optim.GlobalOptimManager.get_instance().register_module_override(
                getattr(embeddings, attr), "weight", {"optim_bits": optim_bits}
            )

model = AutoModel.from_pretrained("microsoft/deberta-v3-base")
trainable_params = [p for p in model.parameters() if p.requires_grad]

optimizer = bnb.optim.AdamW(trainable_params, lr=2e-5, optim_bits=8)
set_embedding_parameters_bits(model.embeddings)
```

## Workflow

1. Install bitsandbytes (`pip install bitsandbytes`)
2. Replace `torch.optim.AdamW` with `bnb.optim.AdamW(optim_bits=8)`
3. Register 32-bit overrides for embedding submodules via `GlobalOptimManager`
4. Train normally — optimizer automatically handles quantized state updates

## Key Decisions

- **Which layers to override**: Embeddings are critical; other layers tolerate 8-bit well
- **Memory savings**: ~75% optimizer memory reduction (32-bit → 8-bit states)
- **Compatibility**: Works with any PyTorch model, not just HuggingFace
- **Combine with**: Layer freezing + gradient checkpointing for maximum memory efficiency

## References

- [Optimization approaches for Transformers](https://www.kaggle.com/code/vad13irt/optimization-approaches-for-transformers)
