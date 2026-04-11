---
name: nlp-yes-no-answer-type-routing
description: >
  Routes QA predictions through an answer-type classifier to emit boolean answers, extractive spans, or null based on type logits.
---
# Yes/No Answer Type Routing

## Overview

Not all questions have extractive answers — some require "yes" or "no," others are unanswerable. Answer type routing uses a classifier head (typically on the CLS token) to predict the answer type, then routes to the appropriate output: return the extracted span for SHORT/LONG types, return "YES"/"NO" for boolean types, or return empty for UNKNOWN. This prevents the model from forcing a span extraction when the answer is boolean or null.

## Quick Start

```python
import numpy as np

ANSWER_TYPES = {0: "UNKNOWN", 1: "YES", 2: "NO", 3: "SHORT", 4: "LONG"}

def route_answer(type_logits, short_spans, short_score_threshold=1.5):
    """Route QA output based on answer type prediction.

    Args:
        type_logits: (5,) logits from answer type head
        short_spans: list of (start, end, score) from span extraction
        short_score_threshold: minimum span score to emit
    Returns:
        dict with answer_type, short_answer, long_answer
    """
    answer_type = ANSWER_TYPES[int(np.argmax(type_logits))]

    if answer_type == "UNKNOWN":
        return {"type": "UNKNOWN", "short": "", "long": ""}
    elif answer_type in ("YES", "NO"):
        return {"type": answer_type, "short": answer_type, "long": ""}
    elif answer_type in ("SHORT", "LONG") and short_spans:
        best = max(short_spans, key=lambda x: x[2])
        if best[2] >= short_score_threshold:
            return {
                "type": answer_type,
                "short": f"{best[0]}:{best[1]}",
                "long": "",  # derive from short via promotion
            }
    return {"type": "UNKNOWN", "short": "", "long": ""}
```

## Workflow

1. Run joint model to get span logits + answer type logits
2. Take argmax of type logits to determine answer category
3. If UNKNOWN: return empty (question is unanswerable)
4. If YES/NO: return the boolean string directly
5. If SHORT/LONG: extract span from span logits, apply score threshold

## Key Decisions

- **Type classes**: 5-class (UNKNOWN/YES/NO/SHORT/LONG) is standard for Natural Questions
- **Score threshold**: Tune on validation; acts as a second filter beyond type classification
- **Fallback**: If type says SHORT but no span exceeds threshold, fall back to UNKNOWN
- **Joint vs pipeline**: Joint training (shared backbone) outperforms separate type classifier

## References

- [TensorFlow 2.0 - Bert YES/NO Answers](https://www.kaggle.com/code/mmmarchetti/tensorflow-2-0-bert-yes-no-answers)
- [BERT Joint Baseline Notebook](https://www.kaggle.com/code/prokaj/bert-joint-baseline-notebook)
