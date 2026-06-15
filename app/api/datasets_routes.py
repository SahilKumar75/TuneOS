"""Dataset search, preview, and generation API routes."""

from __future__ import annotations

import json
import os
import re
import uuid

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from app.api.deps import DATASET_DIR
from app.api.schemas import DatasetGenRequest

router = APIRouter()


@router.get("/datasets/search")
async def search_datasets(q: str = Query(default="", description="Search query")):
    """Search HF Hub datasets."""
    import asyncio

    def _search():
        from huggingface_hub import list_datasets

        results = list(list_datasets(search=q or None, limit=20, sort="downloads"))
        return [
            {
                "id": d.id,
                "downloads": getattr(d, "downloads", 0),
                "tags": getattr(d, "tags", []),
                "description": "",
            }
            for d in results
        ]

    try:
        results = await asyncio.get_event_loop().run_in_executor(None, _search)
        return {"results": results}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/datasets/download")
async def download_dataset_file(
    path: str = Query(..., description="Absolute filesystem path to the file"),
):
    """Serve an alternate-format export (alpaca_json / sharegpt_json) for download."""
    import pathlib

    p = pathlib.Path(path)
    if not p.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    # Restrict to DATASET_DIR to prevent path traversal
    try:
        p.resolve().relative_to(pathlib.Path(DATASET_DIR).resolve())
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Access denied") from exc
    return FileResponse(str(p), filename=p.name, media_type="application/octet-stream")


@router.get("/datasets/{dataset_id:path}/preview")
async def preview_dataset(dataset_id: str):
    """Fetch first 5 rows and column names from an HF Hub dataset."""
    import asyncio

    def _load():
        from datasets import load_dataset

        ds = load_dataset(dataset_id, split="train[:5]", trust_remote_code=False)
        rows = [
            dict(zip(ds.column_names, [ds[col][i] for col in ds.column_names], strict=False))
            for i in range(len(ds))
        ]
        return ds.column_names, rows

    try:
        columns, rows = await asyncio.get_event_loop().run_in_executor(None, _load)
        return {"columns": columns, "rows": rows}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/datasets/generate")
async def generate_dataset(req: DatasetGenRequest):
    """Generate synthetic training data from a plain-English use-case description."""

    async def _generate():
        hf_token = req.hf_token or os.getenv("HF_TOKEN", "")
        openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
        samples = []
        generation_method = "none"
        error_log = []

        # Route to the requested method
        if openrouter_key and req.method == "evol_instruct":
            try:
                samples = await _evol_instruct_generate(
                    req.user_intent, req.n_samples, req.seed_examples, openrouter_key
                )
                generation_method = "evol_instruct"
            except Exception as e:
                error_log.append(f"Evol-Instruct failed: {str(e)}")

        elif openrouter_key and req.method == "persona":
            try:
                samples = await _persona_generate(
                    req.user_intent, req.n_samples, req.personas, openrouter_key
                )
                generation_method = "persona"
            except Exception as e:
                error_log.append(f"Persona generation failed: {str(e)}")

        # Self-instruct / few-shot / auto — try OpenRouter first
        if not samples and openrouter_key and req.method in ("self_instruct", "few_shot", "auto"):
            try:
                samples = await _openrouter_generate(
                    req.user_intent, req.n_samples, req.seed_examples, openrouter_key
                )
                generation_method = "openrouter"
            except Exception as e:
                error_log.append(f"OpenRouter failed: {str(e)}")
                samples = []

        # Fallback to HuggingFace if OpenRouter failed or not available
        if not samples and hf_token and req.method in ("self_instruct", "few_shot"):
            try:
                samples = _self_instruct_generate(
                    req.user_intent, req.n_samples, req.seed_examples, hf_token
                )
                generation_method = "huggingface"
            except Exception as e:
                error_log.append(f"HuggingFace failed: {str(e)}")
                samples = []

        # Final fallback to template generation
        if not samples:
            samples = _template_generate(req.user_intent, req.n_samples)
            generation_method = "template"

        # Dedup by instruction (approximate)
        seen = set()
        unique = []
        for s in samples:
            key = s.get("instruction", "")[:60].lower()
            if key not in seen:
                seen.add(key)
                unique.append(s)

        # Optional LLM-as-judge quality filter
        quality_filtered = False
        if req.quality_threshold > 0 and openrouter_key and unique:
            try:
                before = len(unique)
                unique = await _quality_filter(unique, req.quality_threshold, openrouter_key)
                quality_filtered = True
                error_log_note = f"Quality filter: {before} → {len(unique)} samples"
                error_log.append(error_log_note)
            except Exception as e:
                error_log.append(f"Quality filter failed (skipped): {e}")

        stats = {
            "total_generated": len(samples),
            "final_count": len(unique),
            "diversity_score": _diversity_score(unique),
            "generation_method": generation_method,
            "quality_filtered": quality_filtered,
            "after_quality_filter": len(unique) if quality_filtered else None,
            "errors": error_log if error_log else None,
        }

        # Save canonical JSONL
        os.makedirs(DATASET_DIR, exist_ok=True)
        fname = f"generated_{uuid.uuid4().hex[:8]}.jsonl"
        fpath = os.path.join(DATASET_DIR, fname)
        with open(fpath, "w") as f:
            for row in unique:
                f.write(json.dumps(row) + "\n")

        # Optional alternate format export
        if req.export_format == "alpaca_json":
            alpaca_path = fpath.replace(".jsonl", "_alpaca.json")
            alpaca = [
                {
                    "instruction": r.get("instruction", ""),
                    "input": "",
                    "output": r.get("output", ""),
                }
                for r in unique
            ]
            with open(alpaca_path, "w") as f:
                json.dump(alpaca, f, indent=2)
            stats["alpaca_path"] = alpaca_path
        elif req.export_format == "sharegpt_json":
            sgpt_path = fpath.replace(".jsonl", "_sharegpt.json")
            sgpt = [
                {
                    "conversations": [
                        {"from": "human", "value": r.get("instruction", "")},
                        {"from": "gpt", "value": r.get("output", "")},
                    ]
                }
                for r in unique
            ]
            with open(sgpt_path, "w") as f:
                json.dump(sgpt, f, indent=2)
            stats["sharegpt_path"] = sgpt_path

        return {"samples": unique, "dataset_path": fpath, "stats": stats}

    # Run the async generation
    result = await _generate()
    return result


# ── Dataset generation helpers ───────────────────────────────────


def _self_instruct_generate(intent: str, n: int, seeds: list[dict], hf_token: str) -> list[dict]:
    from huggingface_hub import InferenceClient

    client = InferenceClient(token=hf_token)
    seed_str = "\n".join(
        f"- Instruction: {s['instruction']}\n  Output: {s['output']}"
        for s in (seeds or _default_seeds(intent))[:5]
    )
    prompt = (
        f"You are a dataset creator. The user wants to fine-tune a language model for: {intent}\n\n"
        f"Here are some example instruction/output pairs:\n{seed_str}\n\n"
        f"Generate {n} more diverse and high-quality examples in this JSON format:\n"
        f'[{{"instruction": "...", "output": "..."}}, ...]\n\n'
        f"Return ONLY the JSON array, no other text."
    )
    text = client.text_generation(
        prompt, model="mistralai/Mistral-7B-Instruct-v0.2", max_new_tokens=min(4096, n * 60)
    )
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return []


async def _openrouter_generate(intent: str, n: int, seeds: list[dict], api_key: str) -> list[dict]:
    """Generate synthetic data using OpenRouter API."""
    import httpx

    seed_str = "\n".join(
        f"- Instruction: {s['instruction']}\n  Output: {s['output']}"
        for s in (seeds or _default_seeds(intent))[:5]
    )

    prompt = (
        f"You are a dataset creator. Generate {n} diverse, high-quality instruction/output pairs for fine-tuning a language model.\n\n"
        f"User Intent: {intent}\n\n"
        f"Example pairs:\n{seed_str}\n\n"
        f"Generate {n} NEW examples (not repeating the seeds) in this JSON format:\n"
        f'[{{"instruction": "...", "output": "..."}}, ...]\n\n'
        f"Make the instructions diverse, covering different aspects of the intent.\n"
        f"Return ONLY the JSON array, no markdown, no explanations."
    )

    async with httpx.AsyncClient(timeout=60.0) as http:
        resp = await http.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "X-Title": "TuneOS Dataset Generation",
            },
            json={
                "model": "deepseek/deepseek-v4-flash:free",
                "messages": [
                    {
                        "role": "system",
                        "content": "You generate JSON datasets only. Never use markdown formatting.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": min(8000, n * 150),
                "temperature": 0.8,
            },
        )

        if resp.status_code != 200:
            raise Exception(f"OpenRouter API returned {resp.status_code}: {resp.text}")

        data = resp.json()
        content = data["choices"][0]["message"]["content"]

        # Extract JSON from response
        import re

        # Remove markdown code blocks if present
        content = re.sub(r"```json\s*|\s*```", "", content)
        json_match = re.search(r"\[.*?\]", content, re.DOTALL)
        if json_match:
            samples = json.loads(json_match.group())
            # Validate structure
            valid_samples = []
            for s in samples:
                if isinstance(s, dict) and "instruction" in s and "output" in s:
                    valid_samples.append(s)
            return valid_samples

        raise ValueError("No valid JSON array found in response")


def _default_seeds(intent: str) -> list[dict]:
    intent_lower = intent.lower()
    if any(k in intent_lower for k in ["health", "medical", "doctor", "diabetes"]):
        return [
            {
                "instruction": "What are the symptoms of Type 2 diabetes?",
                "output": "Common symptoms include frequent urination, increased thirst, fatigue, blurred vision, and slow-healing wounds.",
            },
            {
                "instruction": "How often should a diabetic check their blood sugar?",
                "output": "Most people with Type 2 diabetes should check 1–4 times daily, but your doctor will give specific guidance based on your treatment plan.",
            },
        ]
    if any(k in intent_lower for k in ["code", "programming", "python", "developer"]):
        return [
            {
                "instruction": "Write a Python function to reverse a string.",
                "output": "def reverse_string(s: str) -> str:\n    return s[::-1]",
            },
            {
                "instruction": "What is the difference between a list and a tuple in Python?",
                "output": "Lists are mutable (can be changed after creation) while tuples are immutable. Lists use [], tuples use ().",
            },
        ]
    return [
        {
            "instruction": f"Tell me about {intent}.",
            "output": f"Here is a helpful response about {intent}.",
        },
        {
            "instruction": f"What is the best way to approach {intent}?",
            "output": f"The best approach for {intent} involves careful planning, clear goals, and iterative improvement.",
        },
    ]


def _template_generate(intent: str, n: int) -> list[dict]:
    import random

    intent_lower = intent.lower()
    if any(k in intent_lower for k in ["health", "medical", "doctor", "diabetes", "nutrition"]):
        templates = [
            ("What is {topic}?", "It is a medical condition/concept related to {intent_short}."),
            (
                "How can I manage {topic}?",
                "Managing {topic} involves lifestyle changes, medication, and regular monitoring.",
            ),
            (
                "What foods should I avoid with {topic}?",
                "With {topic}, it is best to limit processed foods, refined sugars, and high-sodium items.",
            ),
            (
                "When should I see a doctor about {topic}?",
                "Seek medical advice if you experience persistent or worsening symptoms related to {topic}.",
            ),
            (
                "What are the early signs of {topic}?",
                "Early signs may include fatigue, discomfort, and changes in normal bodily functions.",
            ),
        ]
        topics = [
            "diabetes",
            "hypertension",
            "heart disease",
            "obesity",
            "cholesterol",
            "inflammation",
            "nutrition",
            "exercise recovery",
        ]
    elif any(k in intent_lower for k in ["code", "programming", "python", "developer", "software"]):
        templates = [
            (
                "How do I {topic} in Python?",
                "Here is a simple example:\n```python\n# {topic} example\nresult = None  # implement here\n```",
            ),
            (
                "What is the difference between {topic} and its alternative?",
                "{topic} is commonly used for one scenario while its alternative suits another use case.",
            ),
            (
                "Debug this Python error: {topic}",
                "This error typically occurs when the variable is undefined or out of scope. Check your variable declarations.",
            ),
            (
                "Explain {topic} with an example.",
                "{topic} is a programming concept. Here is a simple example to illustrate it.",
            ),
            (
                "Write a function that {topic}.",
                "```python\ndef solution():\n    # {topic}\n    pass\n```",
            ),
        ]
        topics = [
            "sorts a list",
            "reads a file",
            "handles exceptions",
            "makes an API call",
            "parses JSON",
            "validates input",
            "formats strings",
            "uses decorators",
        ]
    else:
        templates = [
            ("What is {topic}?", "{topic} is an important aspect of {intent_short}."),
            (
                "How does {topic} work?",
                "{topic} works by following a structured process aligned with best practices.",
            ),
            (
                "What are the benefits of {topic}?",
                "The main benefits include efficiency, clarity, and improved outcomes.",
            ),
            (
                "Can you explain {topic} in simple terms?",
                "Simply put, {topic} is about achieving a specific goal in a structured way.",
            ),
            (
                "What should I know about {topic}?",
                "Key things to know: it requires preparation, practice, and continuous learning.",
            ),
        ]
        words = [w for w in intent.split() if len(w) > 3]
        topics = words * max(1, n // max(len(words), 1) + 1)

    intent_short = intent[:30] if len(intent) > 30 else intent
    samples = []
    for _i in range(n):
        topic = random.choice(topics)
        tmpl_inst, tmpl_out = random.choice(templates)
        instruction = tmpl_inst.format(topic=topic, intent_short=intent_short)
        output = tmpl_out.format(topic=topic, intent_short=intent_short)
        samples.append({"instruction": instruction, "output": output})

    return samples


async def _evol_instruct_generate(
    intent: str, n: int, seeds: list[dict], api_key: str
) -> list[dict]:
    """WizardLM-style evolution: rewrite seeds into more complex variants."""
    import math

    import httpx

    seeds = seeds or _default_seeds(intent)
    n_per_seed = math.ceil(n / len(seeds))
    results = []

    async with httpx.AsyncClient(timeout=90.0) as http:
        for seed in seeds:
            if len(results) >= n:
                break
            prompt = (
                f"Original instruction: {seed['instruction']}\n"
                f"Original output: {seed['output']}\n\n"
                f"Produce {n_per_seed} more complex and diverse evolved variants as a JSON array. "
                f"Each variant should be related to: {intent}\n"
                f'Return ONLY JSON: [{{"instruction":"...","output":"..."}}]'
            )
            resp = await http.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "X-Title": "TuneOS Evol-Instruct",
                },
                json={
                    "model": "deepseek/deepseek-v4-flash:free",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an instruction evolver. Output only JSON arrays, no markdown.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": n_per_seed * 200,
                    "temperature": 0.9,
                },
            )
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"]
                content = re.sub(r"```json\s*|\s*```", "", content)
                match = re.search(r"\[.*?\]", content, re.DOTALL)
                if match:
                    try:
                        batch = json.loads(match.group())
                        results.extend(
                            s
                            for s in batch
                            if isinstance(s, dict) and "instruction" in s and "output" in s
                        )
                    except json.JSONDecodeError:
                        pass

    return results[:n]


async def _persona_generate(intent: str, n: int, personas: list[str], api_key: str) -> list[dict]:
    """Generate data from multiple persona viewpoints for diversity."""
    import math

    import httpx

    default_personas = ["expert practitioner", "curious student", "skeptical professional"]
    personas = personas or default_personas
    n_per_persona = math.ceil(n / len(personas))
    results = []

    async with httpx.AsyncClient(timeout=90.0) as http:
        for persona in personas:
            if len(results) >= n:
                break
            prompt = (
                f"You are a {persona} interacting with an AI assistant about: {intent}\n\n"
                f"Generate {n_per_persona} realistic instruction/output pairs from this persona's "
                f"perspective — use their vocabulary, expertise level, and concerns.\n"
                f'Return ONLY JSON: [{{"instruction":"...","output":"..."}}]'
            )
            resp = await http.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "X-Title": "TuneOS Persona Generation",
                },
                json={
                    "model": "deepseek/deepseek-v4-flash:free",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You generate JSON datasets only. Never use markdown formatting.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": n_per_persona * 200,
                    "temperature": 0.85,
                },
            )
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"]
                content = re.sub(r"```json\s*|\s*```", "", content)
                match = re.search(r"\[.*?\]", content, re.DOTALL)
                if match:
                    try:
                        batch = json.loads(match.group())
                        results.extend(
                            s
                            for s in batch
                            if isinstance(s, dict) and "instruction" in s and "output" in s
                        )
                    except json.JSONDecodeError:
                        pass

    return results[:n]


async def _quality_filter(samples: list[dict], threshold: float, api_key: str) -> list[dict]:
    """Score samples 1-5 with LLM-as-judge; discard those below threshold."""
    import httpx

    # Chunk to avoid context overflow (max 100 per call)
    chunk_size = 100
    scored: list[tuple[dict, float]] = []

    async with httpx.AsyncClient(timeout=60.0) as http:
        for i in range(0, len(samples), chunk_size):
            chunk = samples[i : i + chunk_size]
            batch_prompt = (
                "Score each instruction/output pair from 1 (poor) to 5 (excellent) "
                "for clarity, relevance, and quality.\n"
                "Return ONLY a JSON array of numbers, one per pair, in order.\n\n"
                + "\n".join(
                    f"{j + 1}. Instruction: {s['instruction'][:200]}\n"
                    f"   Output: {s['output'][:200]}"
                    for j, s in enumerate(chunk)
                )
            )
            resp = await http.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "X-Title": "TuneOS Quality Filter",
                },
                json={
                    "model": "deepseek/deepseek-v4-flash:free",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You rate dataset quality. Respond only with a JSON array of numbers.",
                        },
                        {"role": "user", "content": batch_prompt},
                    ],
                    "max_tokens": len(chunk) * 5,
                    "temperature": 0.0,
                },
            )
            if resp.status_code != 200:
                # If scoring fails for this chunk, keep all samples
                scored.extend((s, 5.0) for s in chunk)
                continue

            content = resp.json()["choices"][0]["message"]["content"]
            match = re.search(r"\[.*?\]", content, re.DOTALL)
            if not match:
                scored.extend((s, 5.0) for s in chunk)
                continue

            try:
                scores = json.loads(match.group())
            except json.JSONDecodeError:
                scored.extend((s, 5.0) for s in chunk)
                continue

            if not isinstance(scores, list) or len(scores) != len(chunk):
                # LLM returned wrong number of scores — keep all samples
                scored.extend((s, 5.0) for s in chunk)
                continue

            for s, score in zip(chunk, scores, strict=True):
                try:
                    scored.append((s, float(score)))
                except (TypeError, ValueError):
                    scored.append((s, 5.0))

    return [s for s, score in scored if score >= threshold]


def _diversity_score(samples: list[dict]) -> float:
    if len(samples) < 2:
        return 0.0
    instructions = [s.get("instruction", "") for s in samples]
    # Approximate diversity: avg fraction of unique words across instructions
    all_words = set()
    per_sample_words = []
    for inst in instructions:
        words = set(inst.lower().split())
        per_sample_words.append(words)
        all_words |= words
    if not all_words:
        return 0.0
    avg_unique = sum(len(w) for w in per_sample_words) / len(per_sample_words)
    return round(min(1.0, avg_unique / max(len(all_words), 1)), 3)
