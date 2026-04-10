---
name: nlp-space-aware-span-postprocessing
description: Clean up character-level span predictions by removing isolated space characters at span boundaries
domain: nlp
---

# Space-Aware Span Post-Processing

## Overview

Tokenizers (especially BPE/RoBERTa) often include leading/trailing spaces in token offsets. This causes span predictions to include stray space characters at boundaries. Post-process by removing isolated space predictions and bridging spaces between positive predictions.

## Quick Start

```python
import numpy as np

def postprocess_spaces(predictions, text):
    """Fix space artifacts in character-level span predictions."""
    preds = np.copy(predictions)
    for i in range(1, len(text) - 1):
        if text[i] == " ":
            if preds[i] and not preds[i - 1]:    # leading space
                preds[i] = 0
            if preds[i] and not preds[i + 1]:    # trailing space
                preds[i] = 0
            if preds[i - 1] and preds[i + 1]:    # bridge gap
                preds[i] = 1
    return preds
```

## Workflow

1. Get binary character-level predictions (after thresholding)
2. Scan for space characters
3. Remove spaces at span boundaries (no positive neighbor on one side)
4. Bridge spaces between two positive neighbors (merge adjacent spans)

## Key Decisions

- **Why needed**: RoBERTa tokenizer includes leading spaces in tokens (`Ġword`), so offset mapping assigns space characters the same probability as the word
- **Bridge rule**: if both neighbors are positive, the space between them should be too — avoids splitting `"chest pain"` into `"chest"` + `"pain"`
- **Apply after thresholding**: operates on binary predictions, not probabilities

## References

- Source: [roberta-strikes-back](https://www.kaggle.com/code/theoviel/roberta-strikes-back)
- Competition: NBME - Score Clinical Patient Notes
