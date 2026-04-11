---
name: nlp-regex-hybrid-ner-fallback
description: >
  Supplements transformer NER predictions with regex-based detection for structured entities (email, phone, URL), aligning regex matches back to token indices via subsequence search.
---
# Regex Hybrid NER Fallback

## Overview

Transformer NER models excel at contextual entity recognition but often miss structured patterns — email addresses, phone numbers, URLs — that follow rigid formats. A regex fallback layer detects these patterns in the raw text, aligns matches back to token indices using subsequence search, converts to BIO labels, and merges with the model's predictions. This hybrid approach typically adds 1-3% F-score by catching entities the model missed, with zero additional model inference cost.

## Quick Start

```python
import re
import spacy

nlp = spacy.blank("en")

EMAIL_RE = re.compile(r'[\w.+-]+@[\w-]+\.[\w.-]+')
PHONE_RE = re.compile(r'\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}')

def find_span(target_tokens, doc_tokens):
    """Find token subsequence in document."""
    spans = []
    for i in range(len(doc_tokens) - len(target_tokens) + 1):
        if doc_tokens[i:i+len(target_tokens)] == target_tokens:
            spans.append(list(range(i, i + len(target_tokens))))
    return spans

def regex_augment(tokens, full_text, doc_id):
    """Detect structured entities via regex and convert to BIO labels."""
    extra_preds = []
    for pattern, label in [(EMAIL_RE, 'EMAIL'), (PHONE_RE, 'PHONE_NUM')]:
        for match in pattern.finditer(full_text):
            target = [t.text for t in nlp.tokenizer(match.group())]
            for span in find_span(target, tokens):
                for i, token_idx in enumerate(span):
                    prefix = 'B' if i == 0 else 'I'
                    extra_preds.append({
                        'document': doc_id,
                        'token': token_idx,
                        'label': f'{prefix}-{label}',
                    })
    return extra_preds

# Merge with model predictions
all_preds = model_preds + regex_augment(tokens, text, doc_id)
```

## Workflow

1. Define regex patterns for structured entity types
2. Find all regex matches in the full text
3. Tokenize each match with the same tokenizer used for the document
4. Find the matching token subsequence in the document's token list
5. Convert to BIO-tagged predictions and merge with model output

## Key Decisions

- **Pattern priority**: Let regex override model predictions for structured types (higher precision)
- **Tokenizer alignment**: Use the same tokenizer for both match and document to ensure alignment
- **URL patterns**: Be careful with URL regexes — too broad catches non-URLs
- **Dedup**: Deduplicate (doc, token, label) triplets after merging

## References

- [PII Detection with Score 0.967](https://www.kaggle.com/code/startalks/pii-detection-with-score-0-967)
