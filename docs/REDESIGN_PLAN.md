# TuneOS — Full Redesign Plan for Claude Code
> Version: 2.0 | Date: 2026-05-29
> Author: Planning session with Sahil
> Purpose: Step-by-step implementation guide for Claude Code

---

## What This Document Is

A complete, actionable plan to redesign TuneOS from its current state
into the intended product. Every section has exact file paths, what to
change, and what to build. Follow in order.

---

## Current State (What Exists)

The app already has:
- `app/components/sidebar.py` — sidebar with projects, nav items (Search, Datasets, Techniques, TuneOS mobile)
- `app/components/layout.py` — two-panel layout (sidebar + main content)
- `app/pages/landing.py` — main input page (needs to be read, not shown here)
- `app/state/app_state.py` — full state with HF/GitHub API fetching, preview logic, project history
- `app/api.py` — REST API (health, gpu, models, jobs — fully wired to Celery)
- `trainer/` — complete LoRA, QLoRA, finetune, dataset pipeline
- `workers/` — Celery tasks, status via Redis

The state already auto-detects HF vs GitHub from the URL. The API fetch logic works.

---

## What Needs to Change

### Remove
- "Search" nav item from sidebar
- "TuneOS mobile" nav item from sidebar
- "Training" mode dropdown in the input bar (bottom left)
- "Hugging Face / GitHub / Local" source dropdown in the input bar (bottom right)

### Change
- Input bar: single unified input + "+" icon button (no dropdowns)
- Auto-detect source: `huggingface.co` → HF, `github.com` → GitHub, `/path/` or no URL → Local
- Sidebar "New chat" → rename to just show a pencil/plus icon

### Build
See sections below.

---

## Section 1: Sidebar Cleanup

**File:** `app/components/sidebar.py`

**Changes:**
1. Remove `_nav_item("search", "Search")` line
2. Remove `_nav_item("smartphone", "TuneOS mobile")` line
3. In `_collapsed_sidebar()`, remove the search and smartphone icons
4. The remaining nav items should be: New (+), Datasets, Techniques, Settings

**Result:** Clean sidebar with only useful nav items.

---

## Section 2: Input Bar Simplification

**File:** `app/pages/landing.py` (read this file first)

**Goal:** Replace two dropdowns with a single smart input.

**Current bottom bar has:**
- Left: "Training" dropdown (Analytics / Training / Fine-tuning)
- Right: "Hugging Face / GitHub / Local" dropdown

**New bottom bar:**
- Left: "+" icon button (adds project — same as submit)
- Center: Input field (same as now)
- Right: Submit arrow button (same as now)

**Auto-detection logic (already exists in `app_state.py`):**
```python
# Already in AppState.start_project():
if "github.com" in text or self.active_tab == "github":
    preview = await self._fetch_github_preview(text)
else:
    preview = await self._fetch_hf_preview(text)
```

Just extend to handle local paths:
```python
if text.startswith("/") or text.startswith("~/") or text.startswith("./"):
    preview = self._handle_local_path(text)
elif "github.com" in text:
    preview = await self._fetch_github_preview(text)
else:
    preview = await self._fetch_hf_preview(text)
```

Add `_handle_local_path` to `AppState`:
```python
def _handle_local_path(self, path: str) -> dict:
    import os
    name = os.path.basename(path.rstrip("/")) or path
    return {
        "kind": "local",
        "url": path,
        "title": name,
        "owner": "local",
        "summary": f"Local model or dataset at {path}",
        "meta": "Local • No network required",
    }
```

---

## Section 3: Project Workspace (Main Feature)

When user pastes a link and hits enter, the `start_project` event fires,
fetches preview, then `confirm_preview` opens the workspace.

**Currently:** workspace_active = True collapses sidebar, shows something basic.

**New workspace layout** (full screen, sidebar stays collapsed):

```
┌─────────────────────────────────────────────────────────────┐
│  ← Back   [Model Name]   [Source Badge: HF/GitHub/Local]   │
├──────────────────────────┬──────────────────────────────────┤
│                          │                                  │
│   OVERVIEW PANEL (left)  │   CHAT PANEL (right)             │
│                          │                                  │
│   • Model card           │   Chat interface to ask          │
│     - Name               │   questions about the model      │
│     - Source             │   e.g. "What datasets work       │
│     - Tags               │   best for this model?"          │
│     - Downloads/Stars    │                                  │
│     - License            │                                  │
│     - Summary            │                                  │
│                          │                                  │
│   • Dataset section      │                                  │
│     - Linked datasets    │                                  │
│     - Recommended        │                                  │
│     - [Generate] btn     │                                  │
│                          │                                  │
│   • Action Buttons       │                                  │
│     [Train] [Analyze]    │                                  │
│     [Convert] [Notebook] │                                  │
│                          │                                  │
└──────────────────────────┴──────────────────────────────────┘
```

**Files to create/modify:**

1. **`app/pages/workspace.py`** — NEW FILE
   - Contains `workspace_page()` component
   - Shown when `AppState.workspace_active == True`
   - Two-column layout: overview left, chat right

2. **`app/components/overview_panel.py`** — NEW FILE
   - Model card showing all metadata
   - Dataset recommendations section
   - Action buttons row

3. **`app/components/chat_panel.py`** — NEW FILE
   - Simple chat UI (input + message history)
   - Messages stored in state
   - For now: echoes back info about the model (no LLM needed yet)
   - Later: wire to local model

4. **`app/state/app_state.py`** — UPDATE
   - Add `chat_messages: List[dict]` field
   - Add `send_chat_message` event handler
   - Add `_handle_local_path` method

5. **`app/components/layout.py`** — UPDATE
   - When `workspace_active == True`, render workspace instead of landing

---

## Section 4: Dataset Page

**Route:** `/datasets`
**File:** `app/pages/datasets.py` — NEW FILE

**Layout:**
```
┌─────────────────────────────────────────────────────────────┐
│  Datasets                                      [+ Generate] │
├─────────────────────────────────────────────────────────────┤
│  Search datasets...                                         │
├───────────────┬─────────────────────────────────────────────┤
│               │                                             │
│  Categories   │  Dataset cards grid                         │
│  • NLP        │                                             │
│  • Code       │  ┌──────────┐ ┌──────────┐ ┌──────────┐   │
│  • Math       │  │ alpaca   │ │ dolly    │ │ OpenHerm │   │
│  • Science    │  │ 52K rows │ │ 15K rows │ │ 1M rows  │   │
│  • Chat       │  │ [Use]    │ │ [Use]    │ │ [Use]    │   │
│               │  └──────────┘ └──────────┘ └──────────┘   │
│               │                                             │
│               │  [Load more]                                │
└───────────────┴─────────────────────────────────────────────┘
```

**Data source:** HF Datasets API
```
GET https://huggingface.co/api/datasets?sort=downloads&limit=20&full=True
```

**State:** `app/state/dataset_state.py` — NEW FILE
```python
class DatasetState(rx.State):
    datasets: List[dict] = []
    search_query: str = ""
    selected_category: str = "all"
    loading: bool = False

    @rx.event
    async def load_datasets(self):
        # Fetch from HF API
        ...

    @rx.event
    def search(self, query: str):
        self.search_query = query
```

**Generate Dataset button** → opens modal:
- Input: Topic/domain description
- Input: Format (instruction-response, QA, chat)
- Input: Size (100 / 500 / 1000 / custom)
- Source: Local model OR API (OpenAI/Anthropic key)
- [Generate] button → calls `POST /api/generate` (to be built)

---

## Section 5: Techniques Page

**Route:** `/techniques`
**File:** `app/pages/techniques.py` — NEW FILE

**Layout:** Grid of reference cards, like a wiki/docs page.

**Cards to show:**

| Card | Content |
|---|---|
| LoRA | What it is, when to use, rank recommendations |
| QLoRA | 4-bit quantization, memory savings, M1 limitation |
| Full Fine-tuning | When needed, resource requirements |
| Transfer Learning | Building on existing adapters |
| Dataset Formats | JSONL, CSV, instruction-response, chat format |
| Evaluation Metrics | Perplexity, BLEU, what they mean in plain English |
| Model Selection | Which model for which task |
| Training Tips | Learning rate, batch size, epochs guide |

**Each card has:**
- Title + icon
- 2-3 sentence plain English explanation
- "Best for: ..." tag
- "Avoid when: ..." tag
- Link to relevant docs if applicable

**File:** `app/pages/techniques.py`
No state needed — fully static content.

---

## Section 6: Navigation Wiring

**File:** `app/components/sidebar.py`

Wire nav items to routes:
```python
_nav_item("database", "Datasets", on_click=rx.redirect("/datasets"))
_nav_item("flask-conical", "Techniques", on_click=rx.redirect("/techniques"))
```

**File:** `app/app.py` — add new routes:
```python
from app.pages.datasets import datasets_page
from app.pages.techniques import techniques_page

app.add_page(datasets_page, route="/datasets", title="Datasets — TuneOS")
app.add_page(techniques_page, route="/techniques", title="Techniques — TuneOS")
```

---

## Section 7: Training Flow (Existing → Polish)

The training flow already exists as pages. Wire them properly:

**Existing routes:**
- `/upload` — upload dataset
- `/configure` — LoRA config
- `/training` — live training with loss chart
- `/results` — results page

**Action buttons in workspace** should link to these:
```python
rx.button("Train this model", on_click=rx.redirect("/configure"))
```

But first pass the model_id from workspace state to configure page:
- Add `selected_model_id: str` to AppState
- Set it when user is in workspace
- Read it in configure page to pre-fill the model field

---

## Section 8: API Additions Needed

**File:** `app/api.py`

Add these endpoints (stubs first, implement after UI):

```python
@app_api.get("/datasets/search")
async def search_datasets(q: str = "", limit: int = 20):
    """Search HF datasets — proxy to HF API."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"https://huggingface.co/api/datasets",
            params={"search": q, "limit": limit, "sort": "downloads"}
        )
    return resp.json()

@app_api.post("/generate")
async def generate_dataset(config: GenerateConfig):
    """Generate synthetic dataset — enqueue Celery task."""
    job_id = str(uuid.uuid4())
    # TODO: enqueue generate_task
    return {"job_id": job_id, "status": "queued"}

@app_api.get("/recommend/datasets")
async def recommend_datasets(model_id: str):
    """Recommend datasets for a given model based on its tags."""
    # Fetch model tags from HF, match to known good datasets
    return {"recommendations": []}
```

---

## Section 9: M1 / Hardware Backend Detection

**File:** `trainer/config.py` — UPDATE

Add device detection:
```python
import torch
import platform

def get_device() -> str:
    """Detect best available device."""
    if torch.cuda.is_available():
        return "cuda"
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def get_training_mode() -> str:
    """
    Returns 'qlora' for CUDA (4-bit supported),
    'lora' for MPS/CPU (4-bit not supported on Apple Silicon).
    """
    device = get_device()
    if device == "cuda":
        return "qlora"
    return "lora"  # Full-precision LoRA for M1
```

**File:** `trainer/finetune.py` — UPDATE

Use `get_training_mode()` to automatically pick QLoRA vs LoRA:
```python
from trainer.config import get_training_mode

def finetune(...):
    mode = get_training_mode()
    if mode == "qlora":
        model = prepare_qlora_model(model_cfg)
    else:
        model, tokenizer = load_model_and_tokenizer(model_cfg)
        model = inject_lora(model, lora_cfg)
```

Show this in the UI — on the Configure page, display:
```
"Detected: Apple M1 (MPS) — using full-precision LoRA"
"Detected: NVIDIA RTX 3090 — using 4-bit QLoRA"
```

---

## Implementation Order for Claude Code

### Phase 1 — UI Cleanup (do first, fast wins)
1. `sidebar.py` — remove Search + TuneOS mobile
2. `landing.py` — remove dropdowns, add auto-detect
3. `app_state.py` — add `_handle_local_path`
4. Test: run `poetry run python -m reflex run`, verify sidebar looks right

### Phase 2 — Workspace Page
5. `app/pages/workspace.py` — overview + chat two-column layout
6. `app/components/overview_panel.py` — model card + dataset section + action buttons
7. `app/components/chat_panel.py` — chat UI (static for now)
8. `app/state/app_state.py` — add chat messages state
9. `app/components/layout.py` — wire workspace_active → show workspace
10. Test: paste a HF link, verify workspace shows

### Phase 3 — New Pages
11. `app/pages/datasets.py` + `app/state/dataset_state.py`
12. `app/pages/techniques.py` (static content)
13. Wire sidebar nav items to routes
14. Add routes to `app/app.py`

### Phase 4 — Backend Intelligence
15. `trainer/config.py` — device detection
16. `trainer/finetune.py` — auto LoRA vs QLoRA
17. `app/api.py` — dataset search + recommend endpoints
18. Show device info on configure page

### Phase 5 — Polish
19. `app/pages/configure.py` — pre-fill model from workspace state
20. Fix any broken existing pages (upload, training, results)
21. `git add -A && git commit -m "feat: complete UI redesign v2"`

---

## File Map (Complete)

```
app/
├── app.py                    UPDATE — add new routes
├── api.py                    UPDATE — add dataset/generate endpoints
├── styles.py                 NO CHANGE
├── components/
│   ├── sidebar.py            UPDATE — remove Search, TuneOS mobile
│   ├── layout.py             UPDATE — wire workspace
│   ├── overview_panel.py     CREATE
│   ├── chat_panel.py         CREATE
│   ├── loss_chart.py         NO CHANGE (keep for training page)
│   ├── config_form.py        NO CHANGE
│   ├── model_card.py         NO CHANGE
│   └── ...
├── pages/
│   ├── landing.py            UPDATE — simplify input bar
│   ├── workspace.py          CREATE
│   ├── datasets.py           CREATE
│   ├── techniques.py         CREATE
│   ├── configure.py          UPDATE — pre-fill from workspace state
│   ├── training.py           NO CHANGE (keep existing)
│   ├── results.py            NO CHANGE (keep existing)
│   └── upload.py             NO CHANGE (keep existing)
└── state/
    ├── app_state.py          UPDATE — chat messages, local path, device info
    ├── dataset_state.py      CREATE
    ├── job_state.py          NO CHANGE
    └── model_state.py        NO CHANGE
trainer/
├── config.py                 UPDATE — device detection
└── finetune.py               UPDATE — auto backend selection
```

---

## Key Design Decisions

1. **No dropdowns in input bar** — auto-detect from URL/path. Simpler UX.
2. **Workspace is the core experience** — everything happens after adding a project.
3. **Chat panel is passive first** — shows model info answers, no LLM needed yet.
4. **M1 gets full-precision LoRA** — bitsandbytes 4-bit doesn't work on MPS. Show this clearly to user.
5. **Datasets page talks to HF API** — real data, not mocked.
6. **Techniques page is static** — pure reference content, no API needed.
7. **Projects in sidebar are still mocked** — real persistence (DB) is Phase 2 of the product.
