---
name: nlp-cross-model-candidate-pool
description: Merge candidate pools from multiple independent seq2seq models before MBR selection to reduce shared failure modes
domain: nlp
---

# Cross-Model Candidate Pool

## Overview

Single-model MBR candidates share the same biases. Run 2+ independent models (different architectures or checkpoints), merge their candidate pools, then apply MBR on the combined pool. Reduces correlated errors and improves consensus quality.

## Quick Start

```python
def cross_model_mbr(models, tokenizer, input_ids, attn_mask, mbr_fn):
    combined_pool = []
    for model in models:
        candidates = generate_diverse_candidates(model, tokenizer, input_ids, attn_mask)
        combined_pool.extend(candidates)
        # Unload model to free GPU memory
        model.cpu()
        del model
        torch.cuda.empty_cache()
    return mbr_fn(combined_pool)
```

## Key Decisions

- **Sequential GPU loading**: load one model at a time, generate candidates, unload before next
- **Pool diversity > pool size**: 2 different architectures × 10 candidates beats 1 model × 20
- **Architecture variety**: e.g. T5-large + mBART, or different fine-tuning seeds of same arch
- **MBR on combined pool**: the scoring step is O(n^2) so cap total pool at ~50

## References

- Source: [lb-35-9-with-regex-corrections-public-model](https://www.kaggle.com/code/vitorhugobarbedo/lb-35-9-with-regex-corrections-public-model)
- Source: [lb-35-9-ensembling-post-processing-baseline](https://www.kaggle.com/code/giovannyrodrguez/lb-35-9-ensembling-post-processing-baseline)
- Competition: Deep Past Challenge - Translate Akkadian to English
