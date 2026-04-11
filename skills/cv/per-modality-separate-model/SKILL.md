---
name: cv-per-modality-separate-model
description: >
  Trains one specialized model per imaging modality or series type, routing inputs by metadata at inference for modality-specific feature learning.
---
# Per-Modality Separate Model

## Overview

Different imaging modalities (CT vs MRI, or MRI T1 vs T2, or X-ray PA vs lateral) have fundamentally different contrast, resolution, and anatomy visibility. A single model must learn to handle all variations, diluting its capacity. Training separate models per modality lets each specialize — Sagittal T1 learns disc morphology while Axial T2 learns nerve root compression. At inference, series metadata routes each input to the correct model. Predictions are then aggregated per study.

## Quick Start

```python
import timm
import torch.nn as nn

MODALITIES = ['Sagittal T1', 'Sagittal T2/STIR', 'Axial T2']

# Train one model per modality
models = {}
optimizers = {}
for mod in MODALITIES:
    model = timm.create_model('efficientnet_b3', pretrained=True,
                               num_classes=75, in_chans=1)
    models[mod] = model.cuda()
    optimizers[mod] = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop: filter batches by modality
for images, labels, modality in dataloader:
    model = models[modality]
    optimizer = optimizers[modality]
    model.train()
    logits = model(images.cuda())
    loss = criterion(logits, labels.cuda())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Inference: route by series_description
def predict_study(study_series):
    predictions = {}
    for series_desc, images in study_series.items():
        model = models[series_desc]
        model.eval()
        with torch.no_grad():
            predictions[series_desc] = model(images.cuda())
    return aggregate(predictions)
```

## Workflow

1. Group training data by modality/series type using metadata
2. Initialize one model per modality (same or different architectures)
3. Train each model only on its modality's data
4. At inference, read series metadata to route inputs to the correct model
5. Aggregate per-modality predictions at the study level

## Key Decisions

- **Shared vs separate architecture**: Start with same backbone; switch to modality-specific if performance differs
- **Data imbalance**: Some modalities have fewer samples — adjust epochs or use class weights
- **Aggregation**: Average, max, or learned combination of per-modality predictions
- **vs channel stacking**: Separate models use more parameters but specialize better

## References

- [RSNA EfficientNet Starter Notebook](https://www.kaggle.com/code/shubhamcodez/rsna-efficientnet-starter-notebook)
