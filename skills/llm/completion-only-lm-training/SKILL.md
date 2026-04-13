---
name: llm-completion-only-lm-training
description: Fine-tune an instruct LLM on a single text field containing prompt+answer using TRL's DataCollatorForCompletionOnlyLM, which masks the loss to only the answer tokens by detecting a response template string at collation time
---

## Overview

The cleanest way to LoRA-fine-tune an instruct model on labeled data is to put the entire `prompt + answer` into a single `text` column and let TRL's `DataCollatorForCompletionOnlyLM` zero-out the loss labels for everything before a marker string (e.g. `"Answer:"`). You don't manually tokenize, you don't build separate `input_ids`/`labels` tensors, you don't pre-split prompt and target. The collator finds the marker token sequence in each row and sets `labels[:marker_end] = -100`. This guarantees the loss flows *only* through the answer tokens, so the model learns the mapping without being penalized for the (constant) prompt template.

## Quick Start

```python
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig

tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.eos_token

def fmt(row):
    return (f"Rule: {row['rule']}\nSubreddit: {row['subreddit']}\n"
            f"Comment: {row['body']}\nAnswer: {'True' if row['label'] else 'False'}")

train_ds = train_df.assign(text=train_df.apply(fmt, axis=1))[['text']]

collator = DataCollatorForCompletionOnlyLM(
    response_template='Answer:',
    tokenizer=tok,
)

trainer = SFTTrainer(
    model=AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype='bfloat16'),
    train_dataset=train_ds,
    args=SFTConfig(
        output_dir='out', num_train_epochs=2, per_device_train_batch_size=4,
        learning_rate=1e-4, bf16=True, max_seq_length=2048,
        dataset_text_field='text',
    ),
    peft_config=LoraConfig(r=64, lora_alpha=128, target_modules='all-linear', task_type='CAUSAL_LM'),
    data_collator=collator,
    tokenizer=tok,
)
trainer.train()
```

## Workflow

1. Format every example as one string: `"<context>\n<prompt>\n<MARKER>: <answer>"`
2. Pick a `response_template` that is *unambiguous* in your prompt — `"Answer:"` works if the word doesn't appear in the context
3. Hand the formatted dataset (one column, default name `text`) to `SFTTrainer` with `dataset_text_field`
4. Pass the collator — it tokenizes the marker, scans each row's `input_ids`, and masks labels up to the marker
5. Use a LoRA `peft_config` to keep VRAM in check; `target_modules='all-linear'` adapts every linear layer
6. Train 1–3 epochs at `lr=1e-4` (LoRA) or `1e-5` (full fine-tune); more epochs overfit fast

## Key Decisions

- **One `text` field, not separate prompt/completion**: `DataCollatorForCompletionOnlyLM` is the entire reason this works — it removes the manual mask-building boilerplate that gets the labels wrong half the time.
- **Marker must be unique**: if `"Answer:"` appears in a few-shot demo earlier in the prompt, the collator masks the WRONG span. Use `"### Final Answer:"` or similar when in doubt.
- **Tokenize the marker once**: pass it as a string; the collator handles the tokenization and edge cases (BOS, leading space) on its own.
- **`pad_token = eos_token`**: most decoder-only tokenizers don't have a pad token; setting it to eos avoids a warning and is correct because pads are masked anyway.
- **`task_type='CAUSAL_LM'`**: `SEQ_2_SEQ_LM` will silently produce a broken adapter; this is the most common copy-paste bug.
- **Save merged weights only if you're not deploying with vLLM**: vLLM can hot-load the LoRA adapter; merging negates that and breaks quantization (see `vllm-lora-adapter-inference`).

## References

- [Qwen2.5 LoRA finetune baseline (training)](https://www.kaggle.com/competitions/jigsaw-agile-community-rules)
