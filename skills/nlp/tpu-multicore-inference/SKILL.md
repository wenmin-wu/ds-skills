---
name: nlp-tpu-multicore-inference
description: >
  Distributes inference across multiple TPU cores using torch_xla, each core writing a CSV shard, then merges shards via groupby mean.
---
# TPU Multicore Inference

## Overview

A single TPU v3-8 has 8 cores. Running inference on one core wastes 7/8 of available compute. Use `torch_xla` multiprocessing to distribute the test set across all cores with `DistributedSampler`. Each core writes predictions to a separate CSV shard; merge by averaging overlapping IDs (from sampler padding) afterward.

## Quick Start

```python
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from torch.utils.data import DataLoader, DistributedSampler

def _mp_fn(rank, flags):
    device = xm.xla_device()
    model = MyModel().to(device)
    model.load_state_dict(torch.load("model.pt", map_location="cpu"))
    model.eval()

    sampler = DistributedSampler(
        test_dataset, num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(), shuffle=False)
    loader = DataLoader(test_dataset, batch_size=32,
                        sampler=sampler, drop_last=False)

    preds = []
    for batch in loader:
        with torch.no_grad():
            out = model(batch["input_ids"].to(device))
        preds.append(out.cpu())

    df = pd.DataFrame({"id": ids, "pred": torch.cat(preds).numpy()})
    df.to_csv(f"shard_{rank}.csv", index=False)

xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=8, start_method="fork")

# Merge shards — average duplicates from sampler padding
submission = (pd.concat([pd.read_csv(f"shard_{i}.csv") for i in range(8)])
              .groupby("id").mean().reset_index())
```

## Workflow

1. Spawn 8 processes via `xmp.spawn` (one per TPU core)
2. Each process creates a `DistributedSampler` for its slice of the test set
3. Run forward passes on local device, collect predictions
4. Write predictions to a per-rank CSV shard
5. After all processes finish, concatenate shards and average overlapping IDs

## Key Decisions

- **start_method**: Use `"fork"` on TPU VMs; `"spawn"` on Colab
- **drop_last=False**: Keep all test samples; handle padding via groupby mean
- **Shard merging**: `groupby("id").mean()` handles duplicate rows from sampler padding
- **Memory**: Each core loads a full model copy; ensure model fits in per-core HBM

## References

- [[TPU-Inference] Super Fast XLMRoberta](https://www.kaggle.com/code/shonenkov/tpu-inference-super-fast-xlmroberta)
