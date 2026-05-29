---
title: TuneOS
emoji: 🎛️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: true
app_port: 7860
---

# TuneOS

Model operations workspace — discover models, explore datasets, and fine-tune with LoRA/QLoRA.

## Secrets required

Set these in Space Settings → Variables and Secrets:

| Secret | Value |
|---|---|
| `REDIS_URL` | Your Redis Cloud URL (`redis://default:PASSWORD@HOST:PORT`) |
| `HF_TOKEN` | Your Hugging Face token (for gated models) |
