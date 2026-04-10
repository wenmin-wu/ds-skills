---
name: nlp-prompt-text-concatenation
description: >
  Prepends prompt question or full prompt text to input with SEP token for context-aware text evaluation.
---
# Prompt-Text Concatenation

## Overview

When evaluating text quality relative to a prompt (e.g., "how well does this summary capture the source?"), prepend the prompt question or full prompt text to the input, separated by the tokenizer's SEP token. This gives the transformer explicit context about what the text should contain.

## Quick Start

```python
def build_input(row, tokenizer, add_question=True, add_prompt=False):
    parts = []
    if add_question:
        parts.append(row["prompt_question"])
    if add_prompt:
        parts.append(row["prompt_text"])
    parts.append(row["text"])

    sep = f" {tokenizer.sep_token} "
    return sep.join(parts)

# Tokenize with combined input
df["input_text"] = df.apply(lambda r: build_input(r, tokenizer), axis=1)
encodings = tokenizer(df["input_text"].tolist(), truncation=True, max_length=512)
```

## Workflow

1. Decide what context to prepend: question only, full prompt, or both
2. Join with tokenizer's SEP token (not a raw separator)
3. Tokenize the combined string with truncation
4. The model sees prompt context → can judge relevance, not just quality

## Key Decisions

- **Question vs full prompt**: Question is shorter, less truncation risk; full prompt gives richer context
- **SEP token**: Use the model's native separator for proper segment handling
- **Truncation**: Long prompts may push out the actual text — consider head-tail truncation
- **When to use**: Any task where quality is relative to a reference or instruction

## References

- CommonLit - Evaluate Student Summaries (Kaggle)
- Source: [infer-2x-t4](https://www.kaggle.com/code/nbroad/infer-2x-t4)
