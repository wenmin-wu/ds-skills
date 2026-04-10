---
name: llm-threaded-multi-gpu-inference
description: Run multiple LLM inference jobs in parallel using Python threads, each pinned to a separate GPU with staggered starts
domain: llm
---

# Threaded Multi-GPU Inference

## Overview

When you have multiple LLMs to run and multiple GPUs available, parallelize using Python threads with each model pinned to a dedicated GPU. Stagger thread starts by ~10 seconds to avoid simultaneous memory allocation spikes. Collect results into shared DataFrames. Cuts wall-clock time by N for N GPUs.

## Quick Start

```python
import threading
import time
import torch

def run_inference_on_gpu(model_path, gpu_id, data, output_dict, name):
    """Run LLM inference on a specific GPU."""
    device = f"cuda:{gpu_id}"
    model = load_model(model_path, device=device)
    tokenizer = load_tokenizer(model_path)
    
    results = []
    for batch in create_batches(data, batch_size=8):
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                          truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        results.append(outputs.logits.cpu())
    
    output_dict[name] = torch.cat(results)

# Launch parallel inference
results = {}
gpu_assignments = [
    ("path/to/gemma", 0, "gemma"),
    ("path/to/qwen", 1, "qwen"),
]

threads = []
for model_path, gpu_id, name in gpu_assignments:
    t = threading.Thread(
        target=run_inference_on_gpu,
        args=(model_path, gpu_id, test_data, results, name)
    )
    threads.append(t)
    t.start()
    time.sleep(10)  # stagger to avoid OOM

for t in threads:
    t.join()
# results["gemma"], results["qwen"] now available
```

## Key Decisions

- **Threads not processes**: GIL releases during CUDA ops, so threads work fine for GPU inference
- **10s stagger**: prevents simultaneous model loading from exhausting CPU RAM
- **Pin per GPU**: each model gets a dedicated GPU — don't share GPUs between models
- **Shared dict**: thread-safe for writes to different keys; no lock needed

## References

- Source: [ensemble-gemma-qwen-deepseek](https://www.kaggle.com/code/kishanvavdara/ensemble-gemma-qwen-deepseek)
- Competition: MAP - Charting Student Math Misunderstandings
