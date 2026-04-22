---
name: cv-generative-output-numeric-cleaning
description: Clean noisy numeric strings from generative model output by removing invalid characters, fixing malformed floats, and handling multiple decimal points
---

# Generative Output Numeric Cleaning

## Overview

Generative vision-language models (Donut, Pix2Struct, Florence) often produce noisy numeric strings — extra spaces, stray characters, multiple decimal points, or malformed scientific notation. A robust cleaning pipeline strips invalid characters, fixes structural issues (multiple dots, multiple signs), and falls back to zero for unparseable values, preventing downstream errors.

## Quick Start

```python
import re

def clean_numeric(str_list):
    result = []
    for s in str_list:
        s = re.sub(r"\s", "", s)
        dtype = float if "." in s else int
        try:
            result.append(dtype(s))
            continue
        except ValueError:
            pass
        s = re.sub(r"[^0-9.\-eE]", "", s)
        if not s:
            result.append(0)
            continue
        if s.count(".") > 1:
            parts = s.split(".")
            s = parts[0] + "." + "".join(parts[1:])
        if s.count("-") > 1:
            s = "-" + s.replace("-", "")
        try:
            result.append(dtype(s))
        except ValueError:
            result.append(0)
    return result

raw = ["12.5", "1,234.5", " -3..2 ", "abc", "1.2e3"]
clean_numeric(raw)  # [12.5, 1234.5, -3.2, 0, 1200.0]
```

## Workflow

1. Strip whitespace from each string
2. Attempt direct `int()` or `float()` parse — fast path for clean values
3. Remove all non-numeric characters (keep digits, `.`, `-`, `e`, `E`)
4. Fix multiple decimal points by merging after first dot
5. Fix multiple negative signs by keeping only one at the start
6. Return 0 as fallback for unparseable values

## Key Decisions

- **Zero fallback**: safer than NaN for downstream aggregation; adjust per use case
- **Comma handling**: `re.sub` removes commas naturally in the stripping step
- **Scientific notation**: preserve `e`/`E` characters for values like `1.2e3`
- **Type detection**: use `.` presence to choose int vs float

## References

- [donut-infer (LB 0.44) [benetech]](https://www.kaggle.com/code/nbroad/donut-infer-lb-0-44-benetech)
- [Tuned Donut](https://www.kaggle.com/code/cody11null/tuned-donut)
