---
name: cv-classifier-free-guidance-diffusion
description: Manual stable diffusion inference loop with classifier-free guidance that interpolates between unconditional and conditional noise predictions for controllable image generation
---

# Classifier-Free Guidance Diffusion

## Overview

Classifier-free guidance (CFG) steers diffusion model generation toward a text prompt without a separate classifier. At each denoising step, the model predicts noise twice — once conditioned on the prompt, once unconditionally — and the final noise is an extrapolation: `noise = uncond + scale * (cond - uncond)`. Higher guidance scales produce images more faithful to the prompt at the cost of diversity.

## Quick Start

```python
import torch
from torch.cuda.amp import autocast

uncond_input = tokenizer([""], padding="max_length", max_length=77, return_tensors="pt")
text_input = tokenizer([prompt], padding="max_length", max_length=77, truncation=True, return_tensors="pt")
with torch.no_grad():
    uncond_emb = text_encoder(uncond_input.input_ids.cuda())[0]
    text_emb = text_encoder(text_input.input_ids.cuda())[0]
text_embeddings = torch.cat([uncond_emb, text_emb])

latents = torch.randn(1, 4, 64, 64).cuda() * scheduler.init_noise_sigma
for i, t in enumerate(scheduler.timesteps):
    inp = torch.cat([latents] * 2)
    inp = scheduler.scale_model_input(inp, t)
    with torch.no_grad():
        noise_pred = unet(inp, t, encoder_hidden_states=text_embeddings).sample
    uncond, cond = noise_pred.chunk(2)
    noise_pred = uncond + 7.5 * (cond - uncond)
    latents = scheduler.step(noise_pred, t, latents).prev_sample
```

## Workflow

1. Encode empty string and target prompt into text embeddings, concatenate
2. Initialize random latents scaled by the scheduler's initial noise sigma
3. For each timestep: duplicate latents, run UNet to get both unconditional and conditional noise predictions
4. Interpolate: `noise = uncond + guidance_scale * (cond - uncond)`
5. Step the scheduler to update latents
6. Decode final latents through VAE (scale by 1/0.18215 first)

## Key Decisions

- **Guidance scale**: 7.5 is the standard default; 3-5 for more diversity, 10-15 for strict prompt adherence
- **Scheduler**: LMS, DDIM, or Euler-A — LMS is smooth, DDIM allows fewer steps, Euler-A adds stochasticity
- **Inference steps**: 20-50; diminishing returns beyond 50 for most schedulers
- **VAE scaling**: always divide latents by 0.18215 before decoding (SD v1.x convention)

## References

- [Text to Image generation (Stable Diffusion)](https://www.kaggle.com/code/burhanuddinlatsaheb/text-to-image-generation-stable-diffusion)
