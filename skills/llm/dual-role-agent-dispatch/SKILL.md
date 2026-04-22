---
name: llm-dual-role-agent-dispatch
description: Single agent entry point that dispatches between multiple roles (ask/answer/guess) based on turn type, combining heuristic and LLM-based strategies
---

# Dual-Role Agent Dispatch

## Overview

In multi-agent environments where one agent serves multiple roles (questioner, answerer, guesser), use a single dispatch function that routes to different strategies based on the turn type. This allows mixing heuristic logic (structured questions, keyword lookup) with LLM inference (open-ended answering) in one agent, and lazy-loads models only for roles that need them.

## Quick Start

```python
def agent_fn(obs, cfg):
    if obs.turnType == "ask":
        return ask_question(obs)
    elif obs.turnType == "answer":
        return answer_question(obs)
    elif obs.turnType == "guess":
        return make_guess(obs)

def ask_question(obs):
    # Heuristic: structured binary search questions
    return f"Is it a {CATEGORIES[state['cat_idx']]}?"

def answer_question(obs):
    # Quick check: keyword in question?
    if obs.keyword.lower() in obs.questions[-1].lower():
        return "yes"
    # Fall back to LLM
    return llm_yes_no(obs.questions[-1], obs.keyword)

def make_guess(obs):
    # Filter candidates by accumulated answers, pick best
    remaining = filter_candidates(obs)
    return remaining.iloc[0]['keyword'] if len(remaining) > 0 else "unknown"
```

## Workflow

1. Define a single `agent_fn(obs, cfg)` entry point
2. Branch on `obs.turnType` to route to role-specific logic
3. Use heuristic strategies where possible (cheaper, deterministic)
4. Fall back to LLM only when heuristic is insufficient
5. Lazy-load the model on first LLM call to save memory

## Key Decisions

- **Heuristic first**: structured questions and keyword matching don't need LLM inference
- **Lazy loading**: don't load the model until the answerer role is actually invoked
- **Shared state**: use module-level variables or a class to persist state across turns
- **Fallback**: always return a valid response — empty/None can crash the environment

## References

- [Starter Code for Llama 8B LLM - [LB 0.750+]](https://www.kaggle.com/code/cdeotte/starter-code-for-llama-8b-llm-lb-0-750)
- [llama3-8b- LLM20 Questions [LB 750]](https://www.kaggle.com/code/wouldyoujustfocus/llama3-8b-llm20-questions-lb-750)
