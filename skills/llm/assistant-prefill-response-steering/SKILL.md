---
name: llm-assistant-prefill-response-steering
description: >
  Steers LLM output format by injecting a partial assistant response into the chat template before generation, forcing structured output without fine-tuning.
---
# Assistant Prefill Response Steering

## Overview

LLMs often ignore output format instructions, producing preambles, explanations, or inconsistent delimiters. Prefilling the assistant turn with the desired format start (e.g., "Here are the keywords comma separated: ") forces the model to continue in that format. This works with any chat-templated model — inject the partial response, trim the template's trailing EOS tokens so generation continues, then strip the prefill from the decoded output.

## Quick Start

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it")

def generate_with_prefill(prompt, prefill="Here are the keywords: "):
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": prefill},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, return_tensors="pt"
    )
    # Trim trailing EOS tokens added by template
    while input_ids[0, -1] == tokenizer.eos_token_id:
        input_ids = input_ids[:, :-1]

    output_ids = model.generate(
        input_ids,
        max_new_tokens=100,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Extract only the generated part after prefill
    return decoded.split(prefill.strip())[-1].strip()

keywords = generate_with_prefill(
    "Extract patent keywords from: <abstract text>",
    prefill="Keywords: "
)
```

## Workflow

1. Construct chat messages with user prompt and partial assistant response
2. Apply chat template to get input token IDs
3. Trim trailing EOS tokens so the model continues generating
4. Generate with `do_sample=False` for deterministic output
5. Strip the prefill string from decoded output to get clean result

## Key Decisions

- **Prefill content**: Match the exact format you want — "JSON: {" for JSON, "1." for numbered lists
- **EOS trimming**: Essential — without it the model sees a completed turn and generates a new user turn
- **Suppress tokens**: Use `begin_suppress_tokens` to block newlines or special chars in the continuation
- **Fallback**: If output is malformed, retry with a simpler prefill or longer max_new_tokens

## References

- [USPTO LLM Solution [Prompt Eng]](https://www.kaggle.com/code/aerdem4/uspto-llm-solution-prompt-eng)
