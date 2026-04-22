---
name: cv-structured-gt-serialization-tokens
description: Serialize structured ground truth (chart data, table rows, form fields) as special-token-delimited sequences for generative vision-language model training
---

# Structured GT Serialization with Special Tokens

## Overview

When training a generative vision model to extract structured data from images (charts, tables, forms), encode the ground truth as a flat token sequence using custom special tokens as delimiters. Add type tokens, separator tokens, and field boundary tokens to the tokenizer, then serialize each sample's annotations into a string the decoder can learn to produce.

## Quick Start

```python
from transformers import DonutProcessor

PROMPT_TOKEN = "<|PROMPT|>"
X_START, X_END = "<x_start>", "<x_end>"
Y_START, Y_END = "<y_start>", "<y_end>"
TYPE_TOKENS = ["<line>", "<bar>", "<scatter>", "<dot>"]

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
processor.tokenizer.add_tokens([PROMPT_TOKEN, X_START, X_END, Y_START, Y_END] + TYPE_TOKENS)

def serialize_gt(chart_type, x_values, y_values):
    type_token = f"<{chart_type}>"
    x_str = X_START + ";".join(map(str, x_values)) + X_END
    y_str = Y_START + ";".join(map(str, y_values)) + Y_END
    return PROMPT_TOKEN + type_token + x_str + y_str

gt_string = serialize_gt("line", ["2020", "2021", "2022"], [10.5, 15.2, 20.1])
# "<|PROMPT|><line><x_start>2020;2021;2022<x_end><y_start>10.5;15.2;20.1<y_end>"
```

## Workflow

1. Define special tokens for structure boundaries and type labels
2. Add all special tokens to the tokenizer
3. Resize model token embeddings to match: `model.decoder.resize_token_embeddings(len(tokenizer))`
4. Serialize each training sample's annotations into the token format
5. At inference, parse the generated sequence by splitting on special tokens

## Key Decisions

- **Semicolon separator**: within fields, use `;` to delimit list items — avoids ambiguity with commas in numbers
- **Type token first**: placing the type token early lets the model condition all subsequent generation on chart type
- **Resize embeddings**: must call `resize_token_embeddings` after adding tokens
- **Parsing**: at inference, split on boundary tokens and handle missing delimiters gracefully

## References

- [donut-train [benetech]](https://www.kaggle.com/code/nbroad/donut-train-benetech)
- [donut-infer (LB 0.44) [benetech]](https://www.kaggle.com/code/nbroad/donut-infer-lb-0-44-benetech)
