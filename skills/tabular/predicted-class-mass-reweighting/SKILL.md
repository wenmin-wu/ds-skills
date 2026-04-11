---
name: tabular-predicted-class-mass-reweighting
description: >
  Post-hoc rescales ensemble probabilities by the inverse of each class's estimated total mass across the test set, correcting for class imbalance in predictions.
---
# Predicted Class Mass Reweighting

## Overview

Even after ensembling, predicted probabilities can be systematically biased toward the majority class — the model assigns too much total probability mass to class 0 and too little to class 1. This technique estimates each class's total mass across all test predictions (sum of predicted probabilities), then rescales each sample's probabilities by the inverse of its class mass. This shifts the calibration so the predicted class distribution better matches the expected true distribution, improving balanced metrics like balanced log loss.

## Quick Start

```python
import numpy as np

def reweight_by_class_mass(probs):
    """Rescale probabilities by inverse class mass.

    Args:
        probs: (n_samples, n_classes) predicted probabilities
    Returns:
        Reweighted and renormalized probabilities
    """
    class_mass = probs.sum(axis=0)  # total mass per class
    inv_mass = 1.0 / class_mass
    reweighted = probs * inv_mass[np.newaxis, :]
    # Renormalize to sum to 1 per sample
    reweighted /= reweighted.sum(axis=1, keepdims=True)
    return reweighted

# After ensemble averaging
ensemble_probs = np.mean([m.predict_proba(X_test) for m in models], axis=0)
calibrated_probs = reweight_by_class_mass(ensemble_probs)
```

## Workflow

1. Compute ensemble-averaged probabilities for the test set
2. Sum probabilities per class to get estimated class mass
3. Multiply each sample's probability by `1 / class_mass`
4. Renormalize so each row sums to 1

## Key Decisions

- **When to apply**: After ensembling, as a final calibration step
- **Binary case**: For 2-class, simplifies to adjusting the decision boundary
- **vs Platt scaling**: This is non-parametric (no fitting needed); Platt scaling requires a calibration set
- **Assumption**: Assumes the true test distribution is more balanced than model predictions suggest

## References

- [Postprocessin_ Ensemble](https://www.kaggle.com/code/vadimkamaev/postprocessin-ensemble)
