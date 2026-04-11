---
name: cv-quantile-threshold-prevalence-matching
description: >
  Sets the binary classification threshold as a prediction quantile matching the expected positive prevalence rate, avoiding manual threshold tuning.
---
# Quantile Threshold Prevalence Matching

## Overview

When the positive class prevalence is known (e.g., ~2% cancer rate from training data or domain knowledge), the classification threshold can be set as the (1 - prevalence) quantile of test predictions. This automatically adapts to the model's calibration: a well-calibrated model's 98th percentile roughly separates the top 2%. No validation set needed for threshold tuning — useful when the test distribution is expected to match training prevalence.

## Quick Start

```python
import numpy as np
import pandas as pd

def prevalence_threshold(predictions, prevalence_rate=0.02):
    """Set threshold as quantile matching expected positive rate."""
    quantile = 1.0 - prevalence_rate
    threshold = np.quantile(predictions, quantile)
    return threshold

# Aggregate image-level predictions to patient level
pred_df = pd.DataFrame({
    'prediction_id': prediction_ids,
    'cancer_prob': image_predictions,
})
patient_preds = pred_df.groupby('prediction_id')['cancer_prob'].mean()

# Set threshold to match ~2% positive rate
threshold = prevalence_threshold(patient_preds.values, prevalence_rate=0.02)
binary_preds = (patient_preds > threshold).astype(int)

print(f"Threshold: {threshold:.4f}")
print(f"Positive rate: {binary_preds.mean():.3%}")
```

## Workflow

1. Generate soft predictions on the test set
2. Aggregate to the submission level if needed (patient, laterality, etc.)
3. Compute threshold as the (1 - prevalence) quantile
4. Binarize predictions using this threshold
5. Verify the resulting positive rate matches expectations

## Key Decisions

- **Prevalence estimate**: Use training set positive rate, or domain knowledge (e.g., screening cancer rate ~1-2%)
- **Aggregation first**: Aggregate multi-image predictions before thresholding — order matters
- **vs grid search**: Quantile matching requires no validation labels; grid search is better when labels are available
- **Calibration**: Works best when model predictions are roughly calibrated; recalibrate first if not

## References

- [SE-ResNeXt50 Full GPU Decoding](https://www.kaggle.com/code/christofhenkel/se-resnext50-full-gpu-decoding)
- [RSNA Breast Baseline - Faster Inference with Dali](https://www.kaggle.com/code/theoviel/rsna-breast-baseline-faster-inference-with-dali)
