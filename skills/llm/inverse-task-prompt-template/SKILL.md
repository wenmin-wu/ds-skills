---
name: llm-inverse-task-prompt-template
description: Structured prompt template for recovering the instruction that transformed one text into another, with labeled original/rewritten fields and explicit task framing
---

# Inverse Task Prompt Template

## Overview

To recover an unknown prompt from an (original, rewritten) text pair, frame the task explicitly: show the LLM both texts with clear labels, state that one was derived from the other via a prompt, and ask it to infer that prompt. The template structure — labeled fields, task description, output format constraint — is critical for consistent results across different LLMs.

## Quick Start

```python
TEMPLATE = """Below, the `Original Text` has been rewritten into `Rewritten Text`
by an LLM with a certain prompt/instruction. Analyze the differences
and infer the specific prompt that was used.

Original Text:
{original_text}

Rewritten Text:
{rewritten_text}

The prompt was:"""

def build_recovery_prompt(original, rewritten):
    return TEMPLATE.format(
        original_text=original[:1000],
        rewritten_text=rewritten[:1000])

prompt = build_recovery_prompt(row['original_text'], row['rewritten_text'])
output = model.generate(tokenizer(prompt, return_tensors='pt').to(device),
                        max_new_tokens=50, do_sample=False)
recovered = tokenizer.decode(output[0], skip_special_tokens=True)
recovered = recovered.split("The prompt was:")[-1].strip()
```

## Workflow

1. Label both texts clearly (`Original Text:`, `Rewritten Text:`)
2. State the task: one was derived from the other via an LLM prompt
3. Ask the model to analyze differences (style, tone, structure, content)
4. End with a response anchor (`The prompt was:`) to constrain output format
5. Parse only the text after the anchor

## Key Decisions

- **Response anchor**: ending with `The prompt was:` forces concise output without preamble
- **Truncation**: limit input texts to 300-1000 words to fit context window
- **do_sample=False**: greedy decoding for deterministic, focused outputs
- **Model choice**: instruction-tuned models (7B-IT, Mistral-Instruct) follow the template better

## References

- [Prompt Recovery with Gemma - KerasNLP Starter](https://www.kaggle.com/code/awsaf49/prompt-recovery-with-gemma-kerasnlp-starter)
