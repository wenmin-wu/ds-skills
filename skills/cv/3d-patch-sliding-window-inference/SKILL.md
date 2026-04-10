---
name: cv-3d-patch-sliding-window-inference
description: >
  Tiles 3D volumes into overlapping patches for inference and averages overlapping regions to produce seamless predictions.
---
# 3D Patch Sliding-Window Inference

## Overview

Large 3D volumes (CT, cryo-ET, MRI) rarely fit in GPU memory whole. Sliding-window inference tiles the volume into overlapping patches, runs the model on each, then blends overlapping regions via averaging or Gaussian weighting. Eliminates boundary artifacts while keeping memory constant.

## Quick Start

```python
from monai.inferers import SlidingWindowInferer

inferer = SlidingWindowInferer(
    roi_size=(96, 96, 96),
    sw_batch_size=4,
    overlap=0.5,
    mode="gaussian",  # Gaussian weighting reduces edge artifacts
)
output = inferer(inputs=volume_tensor, network=model)
```

## Workflow

1. Choose `roi_size` matching the model's training patch size
2. Set `overlap` (0.25–0.5) — higher overlap = smoother but slower
3. Select blending mode: `"gaussian"` weights center pixels more; `"constant"` averages uniformly
4. Run inferer — it handles tiling, batching, and stitching automatically
5. Post-process the full-resolution prediction volume

## Key Decisions

- **Overlap ratio**: 0.5 is standard; 0.25 acceptable if speed-constrained
- **Blending mode**: Gaussian preferred — reduces checkerboard artifacts at patch boundaries
- **sw_batch_size**: Max patches per forward pass; tune to available VRAM
- **Without MONAI**: Implement manually with `np.lib.stride_tricks` + weighted accumulation buffer

## References

- [Baseline UNet train + submit](https://www.kaggle.com/code/fnands/baseline-unet-train-submit)
- [CZII YOLO11+Unet3D-Monai LB.707](https://www.kaggle.com/code/hideyukizushi/czii-yolo11-unet3d-monai-lb-707)
- [3d-unet using 2d image encoder](https://www.kaggle.com/code/hengck23/3d-unet-using-2d-image-encoder)
