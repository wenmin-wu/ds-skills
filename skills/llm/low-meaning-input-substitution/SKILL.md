---
name: llm-low-meaning-input-substitution
description: Replace the actual input text with a generic low-meaning passage to prevent the LLM from fixating on content specifics, forcing it to focus on stylistic and structural transformation cues
---

# Low-Meaning Input Substitution

## Overview

When asking an LLM to infer what transformation was applied to a text, providing the actual original text can cause the model to describe content differences rather than identify the instruction. Substituting a generic, low-meaning passage as the "original" forces the model to focus on stylistic cues (tone, formality, structure) in the rewritten text, producing better prompt recovery results.

## Quick Start

```python
LOW_MEANING = (
    "Across a spectrum of scenarios, the essence of interaction "
    "and exploration unfolds. The narrative weaves through tales "
    "of adventure and discovery, blending technology and tradition "
    "in a tapestry of human experience."
)

def recover_prompt(rewritten_text, model, tokenizer):
    prompt = f"""Original Text: "{LOW_MEANING}"
Rewritten Text: "{rewritten_text}"
Given the two texts above, the Rewritten Text was created from the
Original Text using an LLM with a certain prompt. Infer the prompt
that was likely used. Output only the prompt, one line."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=60)
    return tokenizer.decode(output[0], skip_special_tokens=True)
```

## Workflow

1. Craft a generic, content-neutral passage (50-100 words, covers multiple themes vaguely)
2. Replace the actual original text with this passage in the prompt template
3. Keep the real rewritten text — the model analyzes its style against the neutral baseline
4. The model focuses on tone, structure, and formatting cues rather than content diffs
5. Post-process the output (truncate to one sentence, strip preamble)

## Key Decisions

- **Why it works**: LLMs default to summarizing content differences; generic input removes that signal
- **Passage design**: should be bland, multi-topic, and grammatically correct — not gibberish
- **When to use**: best when the transformation is stylistic (tone, formality) not content-based
- **Combine with few-shot**: pair with few-shot examples that also use the generic passage

## References

- [Mistral 7B Prompt Recovery Version 2](https://www.kaggle.com/code/richolson/mistral-7b-prompt-recovery-version-2)
