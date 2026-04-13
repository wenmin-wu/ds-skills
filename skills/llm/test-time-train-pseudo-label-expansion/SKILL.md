---
name: llm-test-time-train-pseudo-label-expansion
description: Convert a test.csv with paired positive/negative example columns into a labeled training set at inference time, using the OTHER example as the in-prompt demonstration so the model never sees its own target as a few-shot exemplar
---

## Overview

Some Kaggle test sets ship with `positive_example_1`, `positive_example_2`, `negative_example_1`, `negative_example_2` columns alongside the row to score. These are gold labels — they let you do test-time training (TTT) without external data. The trick is the leakage trap: if you flatten naively and feed each example as both the target AND the in-prompt demo, the model just copies the demo. The fix is to always use the *other* example of the same polarity as the demo (`3 - i` index pairing). You get a clean labeled set the size of `4 * len(test)` and a few-shot prompt with one positive + one negative demo that never overlaps the target row.

## Quick Start

```python
import pandas as pd

rows = []
for _, r in test.iterrows():
    for i in [1, 2]:
        rows.append({
            'body': r[f'positive_example_{i}'],
            'rule': r['rule'],
            'subreddit': r['subreddit'],
            'pos_demo': r[f'positive_example_{3 - i}'],   # the OTHER positive
            'neg_demo': r[f'negative_example_{3 - i}'],   # paired negative
            'label': 1,
        })
        rows.append({
            'body': r[f'negative_example_{i}'],
            'rule': r['rule'],
            'subreddit': r['subreddit'],
            'pos_demo': r[f'positive_example_{3 - i}'],
            'neg_demo': r[f'negative_example_{3 - i}'],
            'label': 0,
        })
ttt_df = pd.DataFrame(rows)   # 4 rows per test row, fully labeled
```

## Workflow

1. For each test row with `positive_example_{1,2}` and `negative_example_{1,2}`, emit 4 labeled rows
2. For each emitted row, set the few-shot demo to the *other* index (`3 - i`) so the target is never in its own prompt
3. Train a classifier (LoRA on an instruct model, or just fit logistic regression on hidden states) on these flattened pseudo-labels
4. At inference on the original test row, build the prompt with `positive_example_1` + `negative_example_1` (either index works since the target row is unlabeled)
5. Optionally combine with public-comp training data — TTT pseudo-labels are domain-perfect but small

## Key Decisions

- **`3 - i` pairing, not random**: random demo selection mixes polarities and adds variance; deterministic pairing is reproducible and leak-free.
- **Both polarities flattened together**: keeps class balance per test row at exactly 50/50, no resampling needed.
- **TTT, not just few-shot**: the labeled rows are real targets — fine-tune (LoRA) on them, don't only use them as prompt context. Few-shot alone leaves capacity on the table.
- **Don't deduplicate across test rows**: the same `(body, rule)` may legitimately repeat with different surrounding context; dedup hurts.
- **Keep `subreddit` and `rule` in the prompt**: the labels are conditional on the rule; stripping it collapses distinct decision boundaries.

## References

- [Test on testdataset (Qwen embedding + Llama + LR)](https://www.kaggle.com/competitions/jigsaw-agile-community-rules)
