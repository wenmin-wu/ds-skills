---
name: nlp-structured-prompt-formatting
description: Format multi-field tabular data into a structured natural language prompt with labeled sections for encoder or LLM classification
domain: nlp
---

# Structured Prompt Formatting

## Overview

When input data has multiple fields (question, answer, explanation, metadata), concatenate them into a structured text prompt with labeled sections. This converts tabular rows into natural language that encoder models (BERT, DeBERTa) or LLMs (Gemma, Qwen) can process. Field labels act as implicit attention anchors, helping the model parse structure.

## Quick Start

```python
def format_prompt(row, fields, separator='\n'):
    """Format a row of data into a labeled text prompt.
    
    Args:
        row: dict-like row (DataFrame row, dict)
        fields: list of (label, column_name) tuples
        separator: delimiter between fields
    """
    parts = []
    for label, col in fields:
        value = row[col]
        if isinstance(value, bool):
            value = "Yes" if value else "No"
        parts.append(f"{label}: {value}")
    return separator.join(parts)

# Define field mapping
fields = [
    ("Question", "QuestionText"),
    ("Answer", "MC_Answer"),
    ("Correct", "is_correct"),
    ("Student Explanation", "StudentExplanation"),
]

df['text'] = df.apply(lambda row: format_prompt(row, fields), axis=1)
# Output: "Question: What is 2+2?\nAnswer: 5\nCorrect: No\nStudent Explanation: ..."
```

## Key Decisions

- **Labeled fields**: "Question:", "Answer:" etc. help the model parse structure vs raw concatenation
- **Newline separator**: mimics natural document structure; use [SEP] for BERT-style models
- **Boolean formatting**: convert True/False to Yes/No for more natural language
- **Field ordering**: put most important fields first — attention patterns favor early tokens

## References

- Source: [gemma2-9b-it-cv-0-945](https://www.kaggle.com/code/cdeotte/gemma2-9b-it-cv-0-945)
- Competition: MAP - Charting Student Math Misunderstandings
