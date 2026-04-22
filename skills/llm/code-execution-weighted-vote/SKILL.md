---
name: llm-code-execution-weighted-vote
description: Execute Python code blocks from LLM math responses in a sandbox, then double-weight code-derived answers in majority voting
---

# Code Execution Weighted Vote

## Overview

Math-reasoning LLMs often generate Python code alongside natural language solutions. Executing this code provides a verified computational answer that should be trusted more than text-only reasoning. Extract Python code blocks, run them in a sandboxed REPL, and add code-derived answers with double weight in the majority vote — boosting the influence of computationally verified results.

## Quick Start

```python
import re
from collections import Counter

def extract_python_blocks(text):
    pattern = r'```python\s*\n(.*?)```'
    return re.findall(pattern, text, re.DOTALL)

def safe_exec(code, timeout=10):
    try:
        local_vars = {}
        exec(code, {"__builtins__": {}}, local_vars)
        output = str(local_vars.get('result', ''))
        nums = re.findall(r'-?\d+', output)
        return int(nums[-1]) if nums else None
    except Exception:
        return None

def weighted_vote(responses, code_weight=2):
    counter = Counter()
    for resp in responses:
        # Text-derived answer (weight=1)
        text_ans = extract_boxed_answer(resp)
        if text_ans is not None:
            counter[text_ans] += 1

        # Code-derived answers (weight=code_weight)
        for code in extract_python_blocks(resp):
            code_ans = safe_exec(code)
            if code_ans is not None:
                counter[code_ans] += code_weight

    return max(counter, key=counter.get) if counter else None
```

## Workflow

1. Generate reasoning responses that may contain Python code blocks
2. Extract \boxed{} answers from text (weight = 1)
3. Extract and execute Python code blocks in sandbox
4. Parse numeric output from code execution (weight = 2)
5. Aggregate all weighted votes via majority voting

## Key Decisions

- **Code weight**: 2x is standard; code is verified computation vs text reasoning
- **Sandbox**: restrict builtins, add timeout — LLM-generated code can be unsafe
- **Multiple code blocks**: a single response may have several; execute all, vote all
- **Execution timeout**: 10-30 seconds per block; kill long-running code

## References

- [[LB 20] QWQ-32B-preview Optimized inference](https://www.kaggle.com/code/mbmmurad/lb-20-qwq-32b-preview-optimized-inference)
