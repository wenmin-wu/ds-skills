---
name: cv-per-patient-slice-maxpool-aggregation
description: Aggregate per-slice CNN predictions into a single patient-level injury score by mean-pooling across TTA copies first, then max-pooling across slices — the worst-slice wins, which matches the medical reality that one bad slice is enough to grade the patient
---

## Overview

For per-patient binary or grade tasks (injury yes/no, severity 0-2-3) where the label depends on *whether the worst slice is bad*, average-pooling across slices washes out the signal: a single high-confidence injury slice gets diluted by 50 healthy ones. The right reduction is max across slices — but only after you've already averaged across TTA / multi-model copies, otherwise you collapse the noise floor. The order matters: `mean(TTA) → max(slices)` is the right composition; `max(everything)` overstates probability and `mean(everything)` understates it.

## Quick Start

```python
import numpy as np

def aggregate_patient(slice_preds, n_tta):
    """
    slice_preds: (n_tta * n_slices, n_outputs) raw model outputs
    Returns: (n_outputs,) patient-level score
    """
    n_slices = slice_preds.shape[0] // n_tta
    pred = slice_preds.reshape(n_tta, n_slices, -1)
    pred = pred.mean(axis=0)              # 1. average TTA copies per slice
    pred = pred.max(axis=0)               # 2. worst slice across the volume
    return pred

patient_scores = np.zeros((len(patient_ids), 11), dtype='float32')
for i, pid in enumerate(patient_ids):
    pdf = test_df.query('patient_id == @pid')
    raw = model.predict(build_dataset(pdf.image_path.tolist()))
    raw = np.concatenate(raw, axis=-1)    # multi-head → flat
    patient_scores[i] = aggregate_patient(raw, n_tta=4)
```

## Workflow

1. Run inference on every slice with N TTA copies (flips, rotations) → `(N * S, C)` predictions
2. Reshape to `(N, S, C)` and average over the TTA axis to denoise per-slice scores
3. Take the max over the slice axis to elevate the worst slice's score to the patient level
4. For multi-head outputs (binary + multi-class), apply this aggregation per head, then concatenate
5. Optionally clip the final scores to `[0.01, 0.99]` if the metric penalizes log-loss extremes

## Key Decisions

- **Mean-then-max, not max-then-mean**: max-then-mean averages already-saturated 1.0s and produces an unstable score; mean-then-max averages noise first then takes a clean argmax.
- **vs. attention pooling**: attention is better in theory but needs training-time supervision; mean→max is zero-parameter and competitive on injury-detection tasks.
- **Don't pool across organs**: the max is per-organ-head; pooling across organs would let a torn liver mask a healthy bowel.
- **Per-series, then per-patient**: if a patient has multiple CT series, max within each series, then *mean* across series — different series are independent observations of the same patient.
- **Stride sampling earlier**: sampling every Nth slice before the model is fine; max-pooling is robust to undersampling as long as N < typical organ-span-in-slices.

## References

- [KerasCV starter notebook (Infer)](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection)
