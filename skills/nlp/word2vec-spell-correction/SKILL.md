---
name: nlp-word2vec-spell-correction
description: >
  Uses Word2Vec vocabulary rank as a word frequency proxy for Norvig-style spell correction, avoiding the need for a separate frequency corpus.
---
# Word2Vec Spell Correction

## Overview

Norvig's spell checker needs word frequencies to pick the most likely correction. If you already have Word2Vec embeddings (e.g., Google News 300d), the vocabulary is sorted by corpus frequency — word rank directly approximates inverse frequency. Use negative rank as the "probability" to select the best candidate from edit-distance neighbors, eliminating the need for a separate word frequency file.

## Quick Start

```python
import gensim

model = gensim.models.KeyedVectors.load_word2vec_format(
    "GoogleNews-vectors-negative300.bin.gz", binary=True)

# Build rank-based "probability" lookup
w_rank = {word: i for i, word in enumerate(model.index_to_key)}

def P(word):
    return -w_rank.get(word, 0)

def edits1(word):
    letters = "abcdefghijklmnopqrstuvwxyz"
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    return set(
        [a + b[1:] for a, b in splits if b] +          # deletes
        [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1] +  # transposes
        [a + c + b[1:] for a, b in splits if b for c in letters] +     # replaces
        [a + c + b for a, b in splits for c in letters]                # inserts
    )

def known(words):
    return {w for w in words if w in w_rank}

def correction(word):
    candidates = known([word]) or known(edits1(word)) or [word]
    return max(candidates, key=P)
```

## Workflow

1. Load a pretrained Word2Vec model (Google News, GloVe converted, etc.)
2. Build a rank dictionary from the vocabulary order
3. Generate edit-distance-1 candidates for each misspelled word
4. Filter to candidates present in the Word2Vec vocabulary
5. Select the candidate with the lowest rank (highest frequency)

## Key Decisions

- **Edit distance**: Distance-1 is fast; chain for distance-2 if recall matters
- **Vocabulary source**: Google News (3M words) has broad coverage; domain models may miss jargon
- **Batch apply**: `df["text"].apply(lambda t: " ".join(correction(w) for w in t.split()))`
- **When to skip**: If using subword tokenizers (BPE, WordPiece), spell correction is less critical

## References

- [Spell Checker using Word2vec](https://www.kaggle.com/code/cpmpml/spell-checker-using-word2vec)
