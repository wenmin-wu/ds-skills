---
name: cv-tensorrt-fp16-compilation
description: >
  Compiles a PyTorch model to a TensorRT FP16 engine via torch_tensorrt for 2-5x inference speedup, saved as reusable TorchScript.
---
# TensorRT FP16 Compilation

## Overview

Kaggle inference kernels have strict time limits (2–9 hours for large datasets). TensorRT compiles a PyTorch model into an optimized GPU engine with FP16 precision, typically achieving 2–5x speedup over vanilla PyTorch. The compiled model is saved as TorchScript and loaded for inference without recompilation. Works with any model that can run a traced forward pass — CNNs, vision transformers, etc.

## Quick Start

```python
import torch
import torch_tensorrt

# Load trained model
model = MyModel()
model.load_state_dict(torch.load('best_model.pth'))
model.eval().cuda().half()

# Merge BatchNorm for better optimization (if applicable)
if hasattr(model, 'merge_bn'):
    model.merge_bn()

# Compile to TensorRT FP16
batch_size = 8
trt_model = torch_tensorrt.compile(
    model,
    inputs=[
        torch_tensorrt.Input(
            [batch_size, 1, 1024, 512],
            dtype=torch.half
        )
    ],
    enabled_precisions={torch.half},
    workspace_size=1 << 32,  # 4GB workspace
    require_full_compilation=True,
)

# Save as TorchScript for reuse
torch.jit.save(trt_model, 'model.trt_fp16.ts')

# Load and run (no recompilation needed)
trt_model = torch.jit.load('model.trt_fp16.ts')
with torch.no_grad():
    output = trt_model(batch.cuda().half())
```

## Workflow

1. Train model in standard PyTorch, save weights
2. Load model, set to eval mode, cast to half precision
3. Compile with `torch_tensorrt.compile()` specifying input shape and FP16
4. Save compiled model as TorchScript (`.ts`)
5. At inference, load TorchScript and run — no recompilation

## Key Decisions

- **Fixed batch size**: TensorRT compiles for a specific batch size — pad the last batch to match
- **workspace_size**: 4GB (`1 << 32`) is safe for most GPUs; reduce for smaller GPUs
- **BN merging**: Merge BatchNorm into Conv before compilation for better optimization
- **Fallback**: If compilation fails on custom ops, use `require_full_compilation=False`

## References

- [3hr TensorRT NextVIT Example](https://www.kaggle.com/code/hengck23/3hr-tensorrt-nextvit-example)
