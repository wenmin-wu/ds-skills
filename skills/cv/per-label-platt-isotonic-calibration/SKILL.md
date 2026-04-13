---
name: cv-per-label-platt-isotonic-calibration
description: Fit a per-label probability calibrator on out-of-fold scores using Platt scaling (logistic regression on raw scores) and fall back to isotonic regression for labels where the logistic doesn't converge — pickle the dict of fitted calibrators and apply at inference for a small but free leaderboard lift on multi-label classification
---

## Overview

Multi-label deep nets are systematically miscalibrated: rare labels are pushed to extreme low probabilities, common labels saturate at high ones. Calibration fixes this without retraining. The recipe: collect out-of-fold scores per label across the training set, fit a `LogisticRegression` per label (Platt scaling), and gracefully fall back to `IsotonicRegression` for labels where Platt fails (constant scores, single-class folds, divergence). Save the dict `{label: ('platt'|'isotonic', model)}` to disk and apply at inference. The lift is usually 0.001-0.003 on macro metrics but it's free, deterministic, and stacks with all other tricks.

## Quick Start

```python
import numpy as np, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

def fit_calibrators(oof_df, gt_df, label_cols):
    cal = {}
    for col in label_cols:
        s = oof_df[col].values
        y = gt_df[col].values
        if np.unique(y).size < 2 or np.allclose(s, s[0]):
            cal[col] = None
            continue
        try:
            lr = LogisticRegression(max_iter=2000)
            lr.fit(s.reshape(-1, 1), y)
            cal[col] = ('platt', lr)
        except Exception:
            iso = IsotonicRegression(out_of_bounds='clip')
            iso.fit(s, y)
            cal[col] = ('isotonic', iso)
    return cal

def apply_calibrators(scores, cal, label_cols):
    out = scores.copy()
    for i, col in enumerate(label_cols):
        c = cal.get(col)
        if c is None: continue
        kind, m = c
        if kind == 'platt':
            out[:, i] = m.predict_proba(scores[:, i].reshape(-1, 1))[:, 1]
        else:
            out[:, i] = m.transform(scores[:, i])
    return out
```

## Workflow

1. Generate out-of-fold predictions from your CV pipeline — never use train predictions
2. For each label column, fit Platt scaling; fall back to isotonic on failure
3. Skip labels with constant scores or single-class targets
4. Pickle the calibrator dict alongside the model checkpoint
5. At inference, transform raw sigmoid scores through the corresponding calibrator
6. Re-tune any per-label thresholds *after* calibration — they shift slightly

## Key Decisions

- **Per-label calibrators, not global**: each label has its own miscalibration curve.
- **Platt before isotonic**: Platt is parametric (1 param), generalizes better with little data; isotonic needs more samples to be stable.
- **Skip degenerate labels gracefully**: a label with no positives in OOF will crash both fitters — return `None` and pass scores through unchanged.
- **Calibrate before threshold tuning**: doing it after invalidates the threshold choices.
- **Out-of-fold is mandatory**: in-fold calibration is overconfident and inflates validation.
- **Lift is small but free**: don't expect leaderboard miracles, but at zero training cost, always worth turning on for the final submission.

## References

- [RSNA Notebook](https://www.kaggle.com/code/luxehadfgsadfg/rsna-notebook)
