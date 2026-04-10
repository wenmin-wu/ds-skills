---
name: llm-synthetic-data-augmentation
description: >
  Generates additional training examples using a stronger LLM (e.g., GPT-3.5) to augment small labeled datasets.
---
# Synthetic Data Augmentation

## Overview

When the labeled training set is tiny (100-500 examples), use a stronger LLM to generate synthetic training data in the same format. For multiple-choice QA, prompt GPT-3.5/4 to create questions with answer options from a knowledge source. This can improve fine-tuned model accuracy by 2-5% with minimal cost.

## Quick Start

```python
from openai import OpenAI

client = OpenAI()

def generate_mcq(topic, n=10):
    prompt = f"""Generate {n} multiple-choice science questions about {topic}.
Format each as:
Question: ...
A) ... B) ... C) ... D) ... E) ...
Answer: X"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
    )
    return parse_mcq(response.choices[0].message.content)

# Combine with real data
synthetic = generate_mcq("physics", n=100)
train_df = pd.concat([real_train_df, synthetic_df]).reset_index(drop=True)
```

## Workflow

1. Analyze the format and distribution of real training data
2. Craft a generation prompt that matches the format exactly
3. Generate synthetic examples across diverse topics
4. Filter: remove duplicates, near-duplicates, and low-quality examples
5. Combine with real data and fine-tune

## Key Decisions

- **Quality vs quantity**: 500 good synthetic examples > 5000 noisy ones
- **Temperature**: 0.7-0.9 for diversity; lower for factual accuracy
- **Filtering**: Deduplicate, validate answer correctness if possible
- **Ratio**: Keep synthetic ≤ 3x real data to avoid distribution shift

## References

- Kaggle LLM Science Exam (Kaggle)
- Source: [new-dataset-deberta-v3-large-training](https://www.kaggle.com/code/radek1/new-dataset-deberta-v3-large-training)
