---
name: nlp-spell-correction-preprocessing
description: >
  Applies domain-aware spelling correction before transformer input to separate spelling errors from content quality.
---
# Spell Correction Preprocessing

## Overview

When evaluating text quality (e.g., student essays), misspellings confuse transformer models that expect well-formed tokens. Autocorrect text before encoding, but augment the spellchecker dictionary with domain vocabulary (e.g., prompt-specific terms) to avoid "correcting" valid domain words. Track misspelling count as a separate feature.

## Quick Start

```python
from autocorrect import Speller
from spellchecker import SpellChecker

class SpellCorrector:
    def __init__(self):
        self.speller = Speller(lang="en")
        self.spellchecker = SpellChecker()

    def add_domain_vocab(self, tokens):
        """Add domain terms so they aren't autocorrected."""
        self.spellchecker.word_frequency.load_words(tokens)
        self.speller.nlp_data.update({t: 1000 for t in tokens})

    def correct(self, text):
        return self.speller(text)

# Usage
corrector = SpellCorrector()
corrector.add_domain_vocab(prompt_tokens)
df["fixed_text"] = df["text"].apply(corrector.correct)
df["misspelling_count"] = df.apply(
    lambda r: len(set(r["text"].split()) - set(r["fixed_text"].split())), axis=1
)
```

## Workflow

1. Extract vocabulary from prompt/reference text
2. Add domain terms to spellchecker dictionary
3. Autocorrect student/generated text
4. Count misspellings as a separate feature for downstream models
5. Feed corrected text to transformer, misspelling count to GBDT

## Key Decisions

- **Domain vocab**: Without it, valid terms get "corrected" to common words
- **Misspelling as feature**: Spelling quality itself is a signal — don't discard it
- **Library choice**: `autocorrect` for correction, `spellchecker` for detection

## References

- CommonLit - Evaluate Student Summaries (Kaggle)
- Source: [tuned-debertav3-lgbm-autocorrect](https://www.kaggle.com/code/cody11null/tuned-debertav3-lgbm-autocorrect)
