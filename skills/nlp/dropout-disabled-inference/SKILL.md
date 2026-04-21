---
name: nlp-dropout-disabled-inference
description: Explicitly zero all dropout probabilities in transformer config at load time for fully deterministic inference
---

# Dropout-Disabled Inference

## Overview

`model.eval()` disables dropout at the module level, but some transformer implementations read dropout rates from the config during forward passes (e.g., custom attention implementations, flash attention paths). Explicitly setting all dropout config values to 0.0 before loading weights guarantees deterministic inference regardless of implementation details.

## Quick Start

```python
from transformers import AutoConfig, AutoModel

config = AutoConfig.from_pretrained('model_path', output_hidden_states=True)
config.hidden_dropout = 0.0
config.hidden_dropout_prob = 0.0
config.attention_dropout = 0.0
config.attention_probs_dropout_prob = 0.0
model = AutoModel.from_pretrained('model_path', config=config)
model.eval()
```

## Workflow

1. Load the model config with `AutoConfig.from_pretrained()`
2. Set all dropout fields to 0.0 (different model families use different field names)
3. Load the model with the modified config
4. Call `model.eval()` as usual

## Key Decisions

- **Which fields**: BERT uses `hidden_dropout_prob` and `attention_probs_dropout_prob`; RoBERTa/DeBERTa may add `hidden_dropout`, `attention_dropout`; check your model's config.json
- **When needed**: always safe to apply; critical when averaging predictions across checkpoints or when reproducibility is required
- **Training vs inference**: only zero dropout at inference — training needs dropout for regularization
- **Alternative**: some use `model.config.update({"hidden_dropout_prob": 0})` post-load, but pre-load is cleaner

## References

- [LECR-stsb_roberta_base](https://www.kaggle.com/code/yuiwai/lecr-stsb-roberta-base)
- [0.459 | Single Model Inference w/ PostProcessing](https://www.kaggle.com/code/karakasatarik/0-459-single-model-inference-w-postprocessing)
