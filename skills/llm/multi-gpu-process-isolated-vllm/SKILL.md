---
name: llm-multi-gpu-process-isolated-vllm
description: Run independent vLLM workers on each GPU by spawning one mp.Process per device and setting CUDA_VISIBLE_DEVICES inside the child before vLLM is imported, sidestepping vLLM's single-instance-per-process limitation
---

## Overview

vLLM holds CUDA context globally inside its process — you cannot create two `LLM(...)` objects in the same Python process and pin them to different GPUs. On Kaggle's 2xT4 / 2xL4 nodes that means half your hardware sits idle unless you go tensor-parallel (which forces both GPUs to host the *same* model, halving max batch). The workaround: `multiprocessing.spawn` one child process per GPU, set `os.environ['CUDA_VISIBLE_DEVICES']` *inside* the child before any `import vllm`, then have each child run `tensor_parallel_size=1` on its own GPU. Gather results through a `Manager().dict()`. This lets you run two *different* models (e.g. Qwen-7B on GPU 0, Llama-8B on GPU 1) or shard a long input list across two copies of the same model.

## Quick Start

```python
import multiprocessing as mp
import os

def worker(gpu_id, prompts, return_dict, model_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    from vllm import LLM, SamplingParams              # import AFTER env var
    llm = LLM(model=model_path, tensor_parallel_size=1,
              gpu_memory_utilization=0.92, max_model_len=4096)
    out = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=4))
    return_dict[gpu_id] = [o.outputs[0].text for o in out]

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    results = manager.dict()
    n = len(all_prompts)
    procs = [
        mp.Process(target=worker, args=(0, all_prompts[:n//2], results, MODEL)),
        mp.Process(target=worker, args=(1, all_prompts[n//2:], results, MODEL)),
    ]
    for p in procs: p.start()
    for p in procs: p.join()
    merged = results[0] + results[1]
```

## Workflow

1. `mp.set_start_method('spawn', force=True)` — fork inherits CUDA context and crashes
2. Define a `worker(gpu_id, ...)` function that sets `CUDA_VISIBLE_DEVICES` *as the first line*, then imports vLLM
3. Split the prompt list across N GPUs; each child sees only its slice
4. Use a `Manager().dict()` keyed by `gpu_id` to collect outputs (regular dicts don't survive process boundaries)
5. `start()` all, then `join()` all — never `start();join()` sequentially or you serialize them
6. Concatenate results in a deterministic order after all processes return

## Key Decisions

- **Set `CUDA_VISIBLE_DEVICES` before `import vllm`, not before `LLM(...)`**: vLLM grabs CUDA at import time on some versions; setting the env later is too late.
- **`spawn`, not `fork`**: fork copies the parent's CUDA state, which is corrupted and segfaults on first GPU op.
- **`tensor_parallel_size=1` per child**: each child sees only one GPU thanks to the env var; setting tp=2 inside a child fails because the other GPU is masked.
- **vs. true tensor parallel**: tp shards one model across both GPUs (lower latency, same throughput); process isolation runs two model copies (higher throughput, can host different models). For batch inference, isolation is almost always faster.
- **`gpu_memory_utilization=0.92`, not 0.95**: leaves headroom for the manager + Python overhead per process.

## References

- [Test on testdataset (Qwen embedding + Llama + LR)](https://www.kaggle.com/competitions/jigsaw-agile-community-rules)
