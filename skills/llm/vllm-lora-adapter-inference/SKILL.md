---
name: llm-vllm-lora-adapter-inference
description: Serve a quantized base LLM with a hot-swappable LoRA adapter under vLLM, enabling prefix caching and tensor parallelism so a single fine-tuned adapter runs at production throughput without merging weights
---

## Overview

After LoRA-fine-tuning a base model, the naive deploy is to merge the adapter into the base and ship the merged checkpoint. That kills two things you actually want on Kaggle: (1) the GPTQ quantization on the base, which merging breaks, and (2) the ability to A/B multiple adapters against the same loaded base. vLLM solves both: load the quantized base once with `enable_lora=True`, then attach the adapter per-request with `LoRARequest`. Prefix caching reuses the system-prompt KV across all rows in a batch, which on a 4k-context classification task is usually a 3–5x throughput win.

## Quick Start

```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(
    model=BASE_MODEL,                  # e.g. Qwen2.5-7B-Instruct-GPTQ-Int4
    quantization='gptq',
    tensor_parallel_size=2,
    enable_lora=True,
    max_lora_rank=64,
    enable_prefix_caching=True,
    max_model_len=4096,
    gpu_memory_utilization=0.92,
)

sampling = SamplingParams(temperature=0, max_tokens=4, logprobs=20)
lora = LoRARequest('default', 1, LORA_PATH)

outputs = llm.generate(prompts, sampling, lora_request=lora)
```

## Workflow

1. Pick a quantized base checkpoint (GPTQ-Int4 / AWQ) — vLLM accepts these directly with `quantization=`
2. Set `max_lora_rank` to match your training rank (default 16 silently truncates)
3. Wrap the adapter directory in a `LoRARequest(name, int_id, path)` — the int_id must be unique per adapter
4. Pass `lora_request=` on each `generate` call (or per-prompt as a list)
5. Enable `enable_prefix_caching=True` so repeated system prompts are KV-cached across the batch
6. Use `tensor_parallel_size=N` for the visible GPU count; vLLM shards the base weights, the adapter is replicated

## Key Decisions

- **Don't merge LoRA into a quantized base**: dequant → merge → requant loses accuracy and breaks GPTQ scales.
- **`max_lora_rank` must be ≥ training rank**: vLLM does not auto-detect; setting it too low is silent corruption.
- **Prefix caching wins big on classification**: identical system prompt + few-shot demos = one-time prefill cost.
- **`max_model_len=4096`, not 32k**: classification prompts rarely need long context; smaller `max_model_len` frees KV cache for larger batch.
- **`temperature=0` + tiny `max_tokens`**: for binary scoring you want one token deterministically; combine with `logprobs` to get class probs without sampling.
- **One LLM instance per process**: vLLM is not thread-safe; for multi-GPU process isolation see `multi-gpu-process-isolated-vllm`.

## References

- [Qwen2.5 LoRA finetune baseline (inference)](https://www.kaggle.com/competitions/jigsaw-agile-community-rules)
