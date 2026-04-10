---
name: llm-multiple-choice-logits-processor
description: >
  Constrains LLM generation to a fixed set of valid choice tokens using a logits processor for structured single-token output.
---
# Multiple-Choice Logits Processor

## Overview

When an LLM must select from a discrete set of options (A/B/C/D, 1-9, yes/no), unconstrained generation can produce invalid answers, explanations, or refusals. A logits processor masks all tokens except the valid choices before sampling, guaranteeing a single valid token output. Works with vLLM, HuggingFace, and other frameworks.

## Quick Start

```python
import vllm
from logits_processor_zoo.vllm import MultipleChoiceLogitsProcessor

llm = vllm.LLM(model="Qwen/Qwen2.5-32B-AWQ", quantization="awq")
tokenizer = llm.get_tokenizer()

params = vllm.SamplingParams(
    n=1, top_k=1, temperature=0, max_tokens=1,
    skip_special_tokens=False,
    logits_processors=[
        MultipleChoiceLogitsProcessor(tokenizer, choices=["1","2","3","4","5"])
    ],
)
responses = llm.generate(prompts, params)
predictions = [r.outputs[0].text.strip() for r in responses]
```

## Workflow

1. Define the valid choice set (letters, digits, words)
2. Initialize `MultipleChoiceLogitsProcessor` with tokenizer and choices
3. Pass as `logits_processors` in sampling params
4. Set `max_tokens=1` — only one token needed
5. Parse the single-token output directly as the answer

## Key Decisions

- **temperature=0**: Greedy decoding for deterministic classification
- **Choice format**: Match what the prompt expects ("A"/"B" vs "1"/"2")
- **Multi-token choices**: For longer options, use constrained decoding (e.g., `outlines`, `guidance`) instead
- **Batch efficiency**: vLLM handles batched generation with logits processors natively

## References

- [Qwen14B_Retrieval_Qwen32B_logits-processor-zoo](https://www.kaggle.com/code/jagatkiran/qwen14b-retrieval-qwen32b-logits-processor-zoo)
- [Eedi Qwen32B vllm with logits-processor-zoo](https://www.kaggle.com/code/aerdem4/eedi-qwen32b-vllm-with-logits-processor-zoo)
