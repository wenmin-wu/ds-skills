---
name: nlp-cp1252-utf8-encoding-normalization
description: >
  Resolves mixed cp1252/utf-8 encoding artifacts in text via round-trip encode/decode with custom error handlers and unidecode normalization.
---
# CP1252-UTF8 Encoding Normalization

## Overview

User-generated text (essays, reviews, web scrapes) often contains mojibake — garbled characters from mismatched encodings (cp1252 bytes interpreted as utf-8 or vice versa). Fix this with a round-trip: raw_unicode_escape → utf-8 → cp1252 → utf-8, using custom error handlers to recover gracefully. Finish with `unidecode` to normalize remaining Unicode to ASCII.

## Quick Start

```python
import codecs
from text_unidecode import unidecode

def replace_encoding_with_utf8(error):
    return (error.object[error.start:error.end].encode("utf-8"), error.end)

def replace_decoding_with_cp1252(error):
    return (error.object[error.start:error.end].decode("cp1252"), error.end)

codecs.register_error("replace_encoding_with_utf8", replace_encoding_with_utf8)
codecs.register_error("replace_decoding_with_cp1252", replace_decoding_with_cp1252)

def resolve_encodings_and_normalize(text: str) -> str:
    text = (
        text.encode("raw_unicode_escape")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
        .encode("cp1252", errors="replace_encoding_with_utf8")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
    )
    return unidecode(text)
```

## Workflow

1. Register custom codec error handlers at module load time
2. Apply `resolve_encodings_and_normalize()` to all text columns before tokenization
3. The function handles: raw Unicode escapes, cp1252↔utf-8 confusion, and Unicode→ASCII

## Key Decisions

- **When to use**: Student essays, web scrapes, legacy databases with mixed encodings
- **unidecode**: Converts accented characters to ASCII; may lose information for non-Latin scripts
- **Apply before tokenization**: Prevents tokenizer from splitting garbled bytes into junk tokens
- **Performance**: Fast enough for millions of rows; register error handlers once

## References

- [feedback_deberta_large_LB0.619](https://www.kaggle.com/code/brandonhu0215/feedback-deberta-large-lb0-619)
- [Huge Ensemble](https://www.kaggle.com/code/thedevastator/huge-ensemble)
