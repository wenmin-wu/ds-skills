---
name: nlp-onnx-mixed-precision-export
description: >
  Exports a HuggingFace transformer to ONNX with dynamic axes, then auto-converts to BF16 mixed precision for 30-200% GPU inference speedup with 2x memory reduction.
---
# ONNX Mixed-Precision Export

## Overview

PyTorch inference is flexible but slow for production. ONNX Runtime with CUDA provides 30-50% speedup from graph optimizations alone. Adding BF16 mixed-precision conversion doubles that — the ONNX Runtime auto-mixed-precision tool profiles each operator and converts safe ones to BF16 while keeping precision-sensitive ops in FP32. Total speedup: 30-200% over PyTorch, with <0.1% accuracy loss. This is the fastest path from a trained HuggingFace model to production inference.

## Quick Start

```python
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from onnxruntime.transformers import auto_mixed_precision as amp

model = AutoModelForTokenClassification.from_pretrained("model_path")
tokenizer = AutoTokenizer.from_pretrained("model_path")
model.eval()

# Step 1: Export to ONNX with dynamic axes
dummy = tokenizer("sample text", return_tensors="pt")
torch.onnx.export(
    model,
    (dummy['input_ids'], dummy['attention_mask']),
    "model.onnx",
    opset_version=14,
    input_names=['input_ids', 'attention_mask'],
    output_names=['logits'],
    dynamic_axes={
        'input_ids': {0: 'batch', 1: 'seq'},
        'attention_mask': {0: 'batch', 1: 'seq'},
        'logits': {0: 'batch', 1: 'seq'},
    }
)

# Step 2: Convert to BF16 mixed precision
amp.auto_convert_mixed_precision_model_path(
    "model.onnx", input_data=dummy,
    output_model_path="model_bf16.onnx",
    provider=['CUDAExecutionProvider'],
    keep_io_types=True
)

# Step 3: Run inference
import onnxruntime as ort
session = ort.InferenceSession("model_bf16.onnx",
    providers=['CUDAExecutionProvider'])
outputs = session.run(None, {
    'input_ids': input_ids.numpy(),
    'attention_mask': attention_mask.numpy()
})
```

## Workflow

1. Export trained model to ONNX with dynamic batch/sequence axes
2. Run auto-mixed-precision to convert safe ops to BF16
3. Load with ONNX Runtime CUDAExecutionProvider
4. Run inference with numpy arrays (no torch needed)

## Key Decisions

- **opset_version**: 14+ for full transformer op support
- **BF16 vs FP16**: BF16 has wider dynamic range, fewer overflow issues; FP16 is faster on older GPUs
- **keep_io_types**: True preserves FP32 input/output for compatibility
- **Batching**: ONNX Runtime handles dynamic batching — process multiple samples in one call

## References

- [0.968 to ONNX 30-200% Speedup PII Inference](https://www.kaggle.com/code/lavrikovav/0-968-to-onnx-30-200-speedup-pii-inference)
