# TuneOS — Deep Feature Planning
> Last updated: 2026-05-29

This document architects the 5 unbuilt features from the 8-feature vision.
Each feature has a clear scope, file plan, and implementation order.

---

## Feature 3 — Scientific Data Generator

**Goal:** Let users generate synthetic training datasets without sourcing real data.

### How it works
User provides:
- A topic/domain (e.g. "medical Q&A", "Python code review")
- A format template (instruction-response, QA, chat)
- A target size (100–10,000 examples)

The generator uses a local model (already loaded) to produce examples,
validates format, deduplicates, and saves to `storage/datasets/`.

### Files to create
```
trainer/
  data_generator.py       # Core generation logic
app/
  pages/generate.py       # UI page
  state/generator_state.py # Reflex state (topic, format, count, progress)
app/api.py                # Add POST /generate endpoint
```

### Implementation order
1. `trainer/data_generator.py` — generation loop with prompt templates
2. `POST /api/generate` endpoint — accepts topic/format/count, returns job_id
3. `app/pages/generate.py` — UI form + live progress
4. Wire into `app/app.py` routing

### Key decisions
- Use the already-loaded base model to generate — no extra deps
- Output as JSONL (instruction/response pairs) compatible with `trainer/dataset.py`
- Cap at 10k examples per run to avoid OOM

---

## Feature 4 — Model Conversion (HF ↔ GGUF ↔ SafeTensors)

**Goal:** Convert trained adapters/merged models between formats for deployment.

### How it works
After training, user picks a saved adapter, selects target format, clicks Convert.
The converter merges LoRA weights into the base model then exports.

### Files to create
```
trainer/
  converter.py            # Conversion logic (HF→SafeTensors, HF→GGUF)
workers/
  convert_task.py         # Celery task wrapping converter
app/
  pages/convert.py        # UI page
app/api.py                # Add POST /convert endpoint
```

### Implementation order
1. `trainer/converter.py`:
   - `merge_and_save_safetensors(adapter_path, base_model, output_dir)` — uses `model.merge_and_unload()` + `safetensors`
   - `export_gguf(model_path, output_path)` — shells out to `llama.cpp/convert.py`
2. `workers/convert_task.py` — Celery task, same pattern as `train_task.py`
3. `POST /api/convert` — accepts adapter_id + format, returns job_id
4. `app/pages/convert.py` — adapter picker + format selector + progress

### Key decisions
- SafeTensors: pure Python, no extra system deps
- GGUF: requires `llama.cpp` to be installed — make it optional, show warning if missing
- Always merge LoRA before converting (adapter alone can't be converted to GGUF)

---

## Feature 5 — Model Analysis

**Goal:** Show quantitative metrics about a trained model so users know if training worked.

### How it works
After training completes, user navigates to `/analysis?job_id=...`.
The page shows loss curve, perplexity on a small eval set, and parameter stats.

### Files to create/update
```
trainer/
  evaluate.py             # IMPLEMENT — currently returns None (see Issue #2)
app/
  pages/analysis.py       # UI page
  components/loss_chart.py # Already exists — verify it works
app/api.py                # Add GET /analysis/{job_id}
```

### Implementation order
1. **Implement `trainer/evaluate.py`** first (Issue #2) — this is the blocker
   ```python
   # Actual perplexity: run model on eval set, compute mean cross-entropy loss
   losses = []
   for batch in eval_dataloader:
       with torch.no_grad():
           out = model(**batch)
           losses.append(out.loss.item())
   perplexity = math.exp(sum(losses) / len(losses))
   ```
2. `GET /api/analysis/{job_id}` — reads loss history from Redis + runs evaluate
3. `app/pages/analysis.py` — loss chart + perplexity card + param count
4. Wire `loss_chart.py` component (already exists) into the page

### Key decisions
- Loss history already streams to Redis via `RedisLossCallback` — just read it
- Run eval on a small subset (max 100 examples) to keep it fast
- Show parameter count from `model.num_parameters()` — free, no inference needed

---

## Feature 6 — Model Understanding

**Goal:** Help users understand what the model has learned — architecture view + tokenization.

### How it works
Two sub-features:
1. **Architecture viewer** — shows layer names, shapes, parameter counts per layer
2. **Tokenizer playground** — user types text, sees token IDs + decoded pieces live

### Files to create
```
app/
  pages/understand.py     # UI page with two tabs
  components/arch_tree.py # Architecture tree component
  components/token_viz.py # Tokenizer visualization component
app/api.py                # Add GET /model/architecture, POST /model/tokenize
```

### Implementation order
1. `GET /api/model/architecture?model_id=...` — returns layer tree JSON
   ```python
   # Walk model.named_modules(), collect name/type/param_count per layer
   ```
2. `POST /api/model/tokenize` — accepts text, returns tokens + ids + decoded
   ```python
   tokens = tokenizer(text, return_tensors="pt")
   decoded = [tokenizer.decode([id]) for id in tokens.input_ids[0]]
   ```
3. `app/pages/understand.py` — two-tab layout
4. Architecture tree: collapsible tree using Reflex components
5. Tokenizer: live text input, updates on keystroke

### Key decisions
- Architecture endpoint loads tokenizer only (fast) — model loading is slow
- Tokenizer playground: debounce 300ms to avoid hammering the API
- No GPU needed for either sub-feature

---

## Feature 7 — Jupyter Notebook Integration

**Goal:** Let users open a Jupyter notebook pre-loaded with their trained adapter.

### How it works
After training, user clicks "Open in Notebook". TuneOS:
1. Starts a Jupyter server (subprocess)
2. Copies a pre-seeded `.ipynb` template into the outputs folder
3. Opens the notebook URL in the embedded browser

### Files to create
```
notebooks/
  tuneos_template.ipynb   # Pre-seeded notebook template
desktop/
  notebook_manager.py     # Starts/stops Jupyter server subprocess
app/
  pages/notebooks.py      # UI page with notebook launcher
app/api.py                # Add POST /notebook/launch
```

### Implementation order
1. `notebooks/tuneos_template.ipynb` — template with cells:
   - Load adapter from `outputs/{job_id}`
   - Run inference on sample prompt
   - Plot loss curve from Redis
2. `desktop/notebook_manager.py` — subprocess manager (same pattern as `process_manager.py`)
3. `POST /api/notebook/launch` — copies template, fills in job_id/adapter_path, starts Jupyter, returns URL
4. `app/pages/notebooks.py` — job picker + Launch button + embedded iframe or link

### Key decisions
- Jupyter is an optional dep (in `pyproject.toml` as optional group)
- Template uses `adapter_path` env var — no hardcoded paths
- Notebook server runs on port 8888, separate from Reflex (3000) and API (8000)

---

## Feature 8 — Transfer Learning

**Goal:** Start a new training run from an existing LoRA adapter instead of the base model.

### How it works
On the Configure page, user can optionally pick an existing adapter as the starting point.
The trainer loads the base model, applies the existing adapter, then runs another LoRA training on top.

### Files to create/update
```
trainer/
  finetune.py             # UPDATE — add `base_adapter_path` param
  lora.py                 # UPDATE — add `load_adapter()` function
app/
  pages/configure.py      # UPDATE — add adapter picker UI
app/api.py                # UPDATE — JobConfig gets optional `base_adapter_id`
```

### Implementation order
1. `trainer/lora.py` — add `load_adapter(model, adapter_path)`:
   ```python
   from peft import PeftModel
   model = PeftModel.from_pretrained(model, adapter_path)
   model = model.merge_and_unload()  # flatten before adding new LoRA
   ```
2. `trainer/finetune.py` — accept optional `base_adapter_path`, call `load_adapter` before `inject_lora`
3. `workers/train_task.py` — pass `base_adapter_path` through
4. `app/api.py` — add `base_adapter_id: str | None = None` to `JobConfig`
5. `app/pages/configure.py` — optional adapter dropdown

### Key decisions
- Merge the base adapter before adding new LoRA (cleaner than stacking adapters)
- If `base_adapter_path` is None, behaviour is identical to current training
- Validate adapter compatibility (same base model) before starting

---

## Build Order (recommended)

| Priority | Feature | Why first |
|---|---|---|
| 1 | Feature 5 — Model Analysis | Unblocks `evaluate.py`, uses existing `loss_chart.py` |
| 2 | Feature 8 — Transfer Learning | Small change to existing pipeline, high value |
| 3 | Feature 3 — Data Generator | Standalone, no deps on other features |
| 4 | Feature 4 — Model Conversion | Needs llama.cpp, more complex |
| 5 | Feature 6 — Model Understanding | UI-heavy, lower backend complexity |
| 6 | Feature 7 — Notebooks | Most isolated, optional dep |
