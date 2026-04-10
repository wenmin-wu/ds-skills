---
name: nlp-chrf-bleu-geometric-mean-metric
description: Geometric mean of chrF and BLEU as a balanced composite translation evaluation metric
domain: nlp
---

# chrF-BLEU Geometric Mean Metric

## Overview

chrF captures character-level quality (good for morphology), BLEU captures word-level n-gram precision. Their geometric mean balances both aspects and is more robust than either alone as a model selection metric during training.

## Quick Start

```python
import sacrebleu
import numpy as np

def chrf_bleu_geomean(predictions, references):
    chrf = sacrebleu.corpus_chrf(predictions, [references], word_order=2)
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    c, b = chrf.score, bleu.score
    geo = (c * b) ** 0.5 if c > 0 and b > 0 else 0.0
    return {"chrf": c, "bleu": b, "geo_mean": geo}

# HuggingFace Trainer integration
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    preds = np.where(preds == -100, tokenizer.pad_token_id, preds)
    labels = np.where(labels == -100, tokenizer.pad_token_id, labels)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    return chrf_bleu_geomean(decoded_preds, decoded_labels)
```

## Key Decisions

- **Geometric not arithmetic mean**: penalizes if either metric is very low
- **chrF++ (word_order=2)**: includes word bigrams, more discriminative than plain chrF
- **Use for model selection**: pick checkpoint with best geo_mean on validation set

## References

- Source: [dpc-starter-train](https://www.kaggle.com/code/takamichitoda/dpc-starter-train)
- Competition: Deep Past Challenge - Translate Akkadian to English
