---
name: llm-4bit-nf4-double-quantization
description: Load large LLMs with 4-bit NF4 quantization and optional double quantization via BitsAndBytes to reduce GPU memory by 4x while preserving inference quality
---

# 4-bit NF4 Double Quantization

## Overview

NormalFloat4 (NF4) quantization maps weights to a 4-bit data type optimized for normally-distributed neural network weights. Double quantization further compresses the quantization constants themselves. Together they reduce a 7B model from ~14GB (fp16) to ~4GB, fitting on a single consumer GPU. Quality loss is minimal for inference tasks like scoring, generation, and classification.

## Quick Start

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.1",
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.1")
```

## Workflow

1. Define `BitsAndBytesConfig` with NF4 quant type and bfloat16 compute dtype
2. Enable `double_quant=True` for additional memory savings (~0.4GB on 7B)
3. Load model with `device_map="auto"` for automatic GPU placement
4. Use normally — all inference ops run in bfloat16, only storage is 4-bit
5. For LoRA fine-tuning, quantized weights stay frozen; adapters train in fp16/bf16

## Key Decisions

- **NF4 vs FP4**: NF4 is better for normally-distributed weights (most LLMs); FP4 for uniform distributions
- **Double quantization**: saves ~0.4GB extra with negligible quality loss — always enable
- **Compute dtype**: bfloat16 is preferred over float16 for numerical stability
- **vs 8-bit**: 4-bit uses half the memory of 8-bit with slightly more quality loss — worth it for 7B+ models on 16GB GPUs

## References

- [Perplexity Baseline Phi-2 / Gemma-7B-IT](https://www.kaggle.com/code/itahiro/perplexity-baseline-phi-2-gemma-7b-it)
