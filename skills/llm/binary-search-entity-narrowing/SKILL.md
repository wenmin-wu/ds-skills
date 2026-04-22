---
name: llm-binary-search-entity-narrowing
description: Hierarchical binary search over entity space by asking category, region, then first-letter questions to narrow candidates before guessing
---

# Binary Search Entity Narrowing

## Overview

When an LLM agent must identify an unknown entity from a large candidate set using yes/no questions, a hierarchical binary search strategy is far more efficient than open-ended questioning. Ask about broad categories first (person/place/thing), then sub-categories (continent/country), then discriminating features (first letter). Each answer halves the remaining candidate space.

## Quick Start

```python
CATEGORIES = ["place", "person", "thing"]
CONTINENTS = ["Europe", "Asia", "Africa", "North America", "South America", "Oceania"]

def ask_question(obs, candidates, cat_guess, con_guess, let_guess,
                 category_yes, continent_yes):
    if cat_guess < len(CATEGORIES) and not category_yes:
        q = f"Is the keyword the name of a {CATEGORIES[cat_guess]}?"
        return q, cat_guess + 1, con_guess, let_guess

    if con_guess < len(CONTINENTS) and not continent_yes:
        cat = category_yes[0] if category_yes else "thing"
        q = f"Is the {cat} located in {CONTINENTS[con_guess]}?"
        return q, cat_guess, con_guess + 1, let_guess

    filtered = apply_filters(candidates, category_yes, continent_yes)
    letters = filtered['first_letter'].value_counts().index.tolist()
    if let_guess < len(letters):
        q = f"Does the keyword begin with the letter {letters[let_guess]}?"
        return q, cat_guess, con_guess, let_guess + 1

    return filtered.sample(1).iloc[0]['keyword'], cat_guess, con_guess, let_guess
```

## Workflow

1. Prepare a candidate keyword list with metadata (category, continent, first letter)
2. Ask category questions first (3 questions max → identifies type)
3. Ask continent/region questions (6 questions max → identifies location)
4. Ask first-letter questions to discriminate remaining candidates
5. When candidate set is small enough, guess directly

## Key Decisions

- **Question order**: broad→narrow maximizes information gain per question
- **Category granularity**: 3-5 top-level categories; more = more questions wasted
- **First letter**: high entropy discriminator — 26 possible values
- **When to guess**: when ≤3 candidates remain, guess the most common one
- **Budget**: 20 questions total — typically 3 category + 6 region + 5 letter + 6 guess attempts

## References

- [Starter Code for Llama 8B LLM - [LB 0.750+]](https://www.kaggle.com/code/cdeotte/starter-code-for-llama-8b-llm-lb-0-750)
