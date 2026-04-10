---
name: llm-agreement-confidence-ensemble
description: Ensemble multi-model LLM predictions using weighted combination of average probability, cross-model agreement ratio, and max confidence
domain: llm
---

# Agreement-Confidence Ensemble

## Overview

When ensembling multiple LLMs that output class probabilities, simple averaging ignores model agreement patterns. Score each candidate class as a weighted blend of: (1) mean probability across models, (2) fraction of models that ranked it in top-K (agreement), and (3) maximum single-model confidence. Agreement acts as a voting signal that breaks ties between similarly-scored classes.

## Quick Start

```python
import numpy as np

def agreement_ensemble(model_probs, class_names, weights=(0.6, 0.3, 0.1), top_k=3):
    """Ensemble predictions from multiple models.
    
    Args:
        model_probs: list of (n_samples, n_classes) arrays
        class_names: list of class name strings
        weights: (avg_prob_weight, agreement_weight, max_conf_weight)
        top_k: number of top classes per model for agreement counting
    Returns:
        list of top-K predicted class names per sample
    """
    n_models = len(model_probs)
    n_samples = model_probs[0].shape[0]
    w_avg, w_agree, w_conf = weights
    
    results = []
    for i in range(n_samples):
        scores = {}
        for c, name in enumerate(class_names):
            avg_prob = np.mean([mp[i, c] for mp in model_probs])
            votes = sum(1 for mp in model_probs if c in np.argsort(-mp[i])[:top_k])
            max_conf = max(mp[i, c] for mp in model_probs)
            scores[name] = w_avg * avg_prob + w_agree * (votes / n_models) + w_conf * max_conf
        ranked = sorted(scores, key=scores.get, reverse=True)
        results.append(ranked[:top_k])
    return results
```

## Key Decisions

- **60/30/10 split**: avg_prob dominates; agreement breaks ties; max_conf is tiebreaker
- **Agreement on top-K**: counts how many models rank a class in their top-K, not just top-1
- **Diverse models**: works best with architecturally different models (encoder vs decoder)
- **Tune weights on OOF**: optimize the three weights on validation data if available

## References

- Source: [ensemble-gemma-qwen-deepseek](https://www.kaggle.com/code/kishanvavdara/ensemble-gemma-qwen-deepseek)
- Competition: MAP - Charting Student Math Misunderstandings
