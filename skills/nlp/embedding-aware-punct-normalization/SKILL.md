---
name: nlp-embedding-aware-punct-normalization
description: Use pretrained embedding vocab to decide which punctuation to keep, split, or remove before tokenization
---

## Overview

Different pretrained embeddings tokenize punctuation differently. GoogleNews has a vector for `&` but not for `?`. GloVe has vectors for `!` and `?`. Instead of applying uniform punctuation rules, query the embedding vocab to decide: (1) keep as a token if it has an embedding, (2) split around characters that act as word boundaries, (3) remove characters with no vector. This routinely pushes text coverage from ~90% to >99%.

## Quick Start

```python
def clean_text(x):
    x = str(x)
    # Split punctuation the embedding treats as word boundaries
    # (so 'foo/bar' becomes two tokens 'foo bar')
    for punct in "/-'":
        x = x.replace(punct, ' ')

    # Keep tokens known to have embeddings (surround with spaces so tokenizer splits them)
    for punct in '&':
        x = x.replace(punct, f' {punct} ')

    # Remove punctuation with no embedding coverage
    strip_chars = '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '\u201c\u201d\u2018\u2019'
    for punct in strip_chars:
        x = x.replace(punct, '')
    return x
```

## Workflow

1. Build a dataset vocab frequency dict and check coverage against the pretrained embedding
2. For each frequent OOV punctuation character, test: does the pretrained index contain it?
3. Partition punctuation into 3 sets: keep-as-token, split-as-separator, remove
4. Apply `clean_text` before tokenization
5. Re-run coverage check — aim for > 99% text coverage

## Key Decisions

- **Per-embedding rules**: GloVe, FastText, and Paragram have different punctuation vocabs. Write one `clean_text` per embedding, or use the intersection.
- **Order matters**: Split separators first, then keep-tokens, then strip. Otherwise you may split characters you wanted to keep.
- **vs. blanket removal**: Removing all punctuation loses real signal (e.g., `&` in `R&D`). Embedding-aware rules preserve it.
- **vs. lowercase**: Lowercase aggressively too — most pretrained vocabs are lowercased except for named-entity embeddings.

## References

- [How to: Preprocessing when using embeddings](https://www.kaggle.com/code/christofhenkel/how-to-preprocessing-when-using-embeddings)
