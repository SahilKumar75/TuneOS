# Adapter Techniques

TuneOS supports six parameter-efficient fine-tuning (PEFT) methods, plus two
alignment/distillation recipes that build on top of them. All six adapters are
registered in `trainer/adapters.py` and selected by name through the wizard.

---

## LoRA — Low-Rank Adaptation

LoRA freezes all pre-trained model weights and injects pairs of trainable low-rank
matrices into each Transformer layer. Only those matrices are updated during training,
which means the number of trainable parameters stays small regardless of the base
model size.

A rank-8 LoRA adapter on a 7B model trains roughly 0.5% of the total parameters while
preserving most of the expressiveness of a full fine-tune.

**Key configuration fields**

| Field | What it controls |
|---|---|
| `lora_rank` (r) | Dimension of the injected matrices. Higher rank gives more capacity but uses more memory. 8-16 is a good starting range. |
| `lora_alpha` | Scaling factor applied to the adapter output. Common convention is to set it to 2× rank. |
| `lora_dropout` | Dropout on the adapter path to reduce overfitting. 0.05 is a reasonable default. |

`target_modules` — the specific projection layers the adapter attaches to — are
auto-detected from the model architecture, so the same configuration works across
Mistral, Llama, Gemma, Phi-3, Falcon, Qwen2, and GPT-NeoX families.

---

## QLoRA — Quantized LoRA

QLoRA compresses the base model to 4-bit NormalFloat (NF4) precision using
bitsandbytes, then trains LoRA adapters on top in full precision (BF16/FP16). The
quantized base occupies roughly a quarter of the VRAM of a full-precision load, making
it possible to fine-tune a 7B model on a consumer GPU with 16 GB VRAM.

QLoRA is the default technique in TuneOS because it offers the best VRAM/accuracy
trade-off for models in the 7B-13B range.

---

## AdaLoRA — Adaptive Rank Allocation

AdaLoRA starts with a higher-rank adapter (2× the target rank) and uses singular
value decomposition to prune unimportant components during training. Layers that
contribute less to the task end up with lower effective rank; layers that matter
retain more capacity. The result is often better quality than a fixed-rank LoRA for
the same parameter budget.

---

## IA3 — Infused Adapter by Inhibiting and Amplifying Inner Activations

IA3 scales the keys, values, and feed-forward activations at each layer with learned
vectors rather than injecting full rank matrices. This produces roughly 10-100x fewer
trainable parameters than LoRA for the same model, at the cost of some accuracy.
Useful when training time or storage is extremely constrained.

---

## Prefix Tuning

Prefix tuning prepends a sequence of learned virtual tokens to the key/value states
at every layer. The base model weights are completely frozen; only the prefix
embeddings are trained. The `lora_rank` field is reused as the number of virtual
prefix tokens.

---

## Prompt Tuning

Prompt tuning learns a small set of soft token embeddings prepended to the input
sequence. It is the most parameter-efficient method in the registry — only the
embedding vectors are trained, and the rest of the model is untouched.
`lora_rank` is reused as the number of virtual prompt tokens.

---

## Technique Comparison

| Technique | Trainable params | VRAM (7B base) | Best use case |
|---|---|---|---|
| QLoRA | ~0.5% | ~16 GB | Default — best memory/accuracy for 7B+ |
| LoRA | ~0.5% | ~28 GB | Lower quantization noise; smaller models |
| AdaLoRA | ~0.5% | ~28 GB | Better quality per parameter on complex tasks |
| IA3 | ~0.01% | ~28 GB | Very constrained memory or storage |
| Prefix Tuning | ~0.1% | ~28 GB | No weight modification needed |
| Prompt Tuning | <0.01% | ~28 GB | Fewest possible parameters |

---

## DPO — Direct Preference Optimization

Standard fine-tuning (SFT) trains a model to imitate reference outputs. DPO trains
on preference data instead — triples of `(prompt, chosen, rejected)` — to push the
model toward responses that humans prefer, without a separate reward model.

DPO reuses the same LoRA adapter machinery. The dataset shape and loss function
differ, but the adapter configuration fields (rank, alpha, dropout) are identical.
Select DPO as the training technique in step 1; step 3 will show the column-mapping
card for your preference dataset.

**Key DPO fields**

| Field | Default | Description |
|---|---|---|
| `beta` | 0.1 | KL-penalty coefficient. Lower values allow larger policy deviations. |
| `max_length` | 1024 | Maximum total sequence length (prompt + response). |
| `max_prompt_length` | 512 | Maximum prompt length. |

---

## Knowledge Distillation

Knowledge distillation fine-tunes a smaller student model to match the output
distribution of a larger teacher model. The student learns from the teacher's soft
probability distribution over tokens (via cross-entropy against the teacher logits)
rather than from hard labels alone.

**Key KD fields**

| Field | Default | Description |
|---|---|---|
| `teacher_model` | — | Hugging Face ID of the teacher model. |
| `temperature` | 2.0 | Softmax temperature applied to teacher logits. Higher values produce softer distributions. |
| `alpha` | 0.5 | Weight on the distillation loss vs. standard cross-entropy. `1.0` is pure distillation; `0.0` is pure SFT. |

---

## Adapter Composition

After training, the wizard's advanced mode lets you stack a second adapter on top of
the trained model using `PeftMixedModel`. This is useful for researcher workflows that
want to layer a general-purpose adapter over a task-specific one, or combine
techniques (e.g. a LoRA base with an IA3 overlay).

Supported overlay techniques: `lora`, `adalora`, `ia3`. Enable via the **Adapter
Composition** section in step 4 (advanced mode).
