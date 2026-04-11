---
name: nlp-subtoken-labeling-strategy
description: >
  Controls whether all subtokens or only the first subtoken of each word receive NER labels during training and inference.
---
# Subtoken Labeling Strategy

## Overview

Transformer tokenizers split words into subword tokens ("playing" → "play", "##ing"). For token classification (NER), you must decide: label all subtokens of a word, or only the first? Labeling all subtokens gives more supervision signal and can improve recall, while first-only is cleaner and avoids label noise from meaningless subword pieces. At inference, always use the first subtoken's prediction to represent the word.

## Quick Start

```python
def align_labels(word_labels, word_ids, label_all_subtokens=True):
    """Align word-level labels to subtoken positions.

    Args:
        word_labels: list of labels, one per word
        word_ids: tokenizer output word_ids(), None for special tokens
        label_all_subtokens: if True, label all subtokens; if False, only first
    """
    label_ids = []
    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)  # ignore in loss
        elif word_idx != previous_word_idx:
            label_ids.append(word_labels[word_idx])  # first subtoken
        else:
            if label_all_subtokens:
                label_ids.append(word_labels[word_idx])  # subsequent subtokens
            else:
                label_ids.append(-100)  # ignore subsequent subtokens
        previous_word_idx = word_idx
    return label_ids

def predict_words(token_preds, word_ids):
    """Map token predictions back to words using first-subtoken strategy."""
    word_preds = []
    previous_word_idx = -1
    for idx, word_idx in enumerate(word_ids):
        if word_idx is not None and word_idx != previous_word_idx:
            word_preds.append(token_preds[idx])
        previous_word_idx = word_idx if word_idx is not None else previous_word_idx
    return word_preds
```

## Workflow

1. Tokenize text with `return_offsets_mapping=True` to get `word_ids()`
2. During training: align word labels to subtokens using chosen strategy
3. Set non-labeled positions to -100 (ignored by CrossEntropyLoss)
4. During inference: take first subtoken prediction per word

## Key Decisions

- **Label all**: Better for recall, more training signal; works well with BIO tagging
- **First only**: Cleaner labels, standard in HuggingFace NER examples
- **Inference**: Always use first-subtoken prediction regardless of training strategy
- **BIO consistency**: If labeling all, keep B- only on first subtoken, I- on the rest

## References

- [PyTorch BigBird NER CV 0.615](https://www.kaggle.com/code/cdeotte/pytorch-bigbird-ner-cv-0-615)
