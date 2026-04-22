---
name: llm-turn-based-prompt-accumulation
description: Rebuild multi-turn conversation context by interleaving user/assistant turns with chat template tokens into a single prompt each call
---

# Turn-Based Prompt Accumulation

## Overview

Causal LLMs without native chat APIs need conversation history manually injected into the prompt. Each turn, rebuild the full prompt by interleaving all previous user and assistant messages with the model's special tokens (e.g., `<|start_header_id|>`, `<|eot_id|>` for Llama). This gives the model full context of the conversation without maintaining KV cache across calls.

## Quick Start

```python
def build_chat_prompt(system_prompt, questions, answers, model_format="llama3"):
    prompt = (f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
              f"{system_prompt}<|eot_id|>")

    for i, (q, a) in enumerate(zip(questions, answers)):
        prompt += (f"<|start_header_id|>assistant<|end_header_id|>\n\n"
                   f"{q}<|eot_id|>")
        prompt += (f"<|start_header_id|>user<|end_header_id|>\n\n"
                   f"{a}<|eot_id|>")

    # Open the next assistant turn
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return prompt

prompt = build_chat_prompt(sys_prompt, obs.questions, obs.answers)
output = model.generate(tokenizer(prompt, return_tensors="pt").to("cuda"),
                        max_new_tokens=50)
```

## Workflow

1. Start with system prompt wrapped in model's special tokens
2. For each previous turn, append assistant message then user response with proper headers
3. End with an open assistant header to prompt the next generation
4. Tokenize the full prompt and generate
5. Truncate output at EOT token to extract clean response

## Key Decisions

- **Token format**: Llama3 uses `<|start_header_id|>role<|end_header_id|>`; Gemma uses `<start_of_turn>model`
- **Rebuild vs append**: rebuilding from scratch each turn is simpler and avoids KV cache issues
- **Context length**: monitor total token count — truncate oldest turns if approaching limit
- **EOT truncation**: find the EOT token in generated output and slice before it

## References

- [llama3-8b- LLM20 Questions [LB 750]](https://www.kaggle.com/code/wouldyoujustfocus/llama3-8b-llm20-questions-lb-750)
- [LLM 20 Questions Starter Notebook](https://www.kaggle.com/code/ryanholbrook/llm-20-questions-starter-notebook)
