import pandas as pd
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizer

# Templates live in a dependency-free module shared with the Reflex app
# (single source of truth — see trainer/prompt_templates.py).
from trainer.prompt_templates import PROMPT_TEMPLATE, PROMPT_TEMPLATES  # noqa: F401


def _format_prompt_prefix(row: dict, template: str = "alpaca") -> str:
    """Return just the instruction part of the formatted text (no output).

    Used to compute prompt token length for label masking — the model should
    only compute loss on the response, not on the repeated instruction.
    """
    tmpl = PROMPT_TEMPLATES.get(template, PROMPT_TEMPLATES["alpaca"])
    prefix_tmpl = tmpl.split("{output}")[0]
    return prefix_tmpl.format(instruction=row.get("instruction", ""))


def _format_prompt_prefix(row: dict, template: str = "alpaca") -> str:
    """Return just the instruction part of the formatted text (no output).

    Used to compute prompt token length for label masking — the model should
    only compute loss on the response, not on the repeated instruction.
    """
    tmpl = PROMPT_TEMPLATES.get(template, PROMPT_TEMPLATES["alpaca"])
    prefix_tmpl = tmpl.split("{output}")[0]
    return prefix_tmpl.format(instruction=row.get("instruction", ""))


def format_prompt(
    row: dict,
    instruction_col: str = "instruction",
    output_col: str = "output",
    template: str = "alpaca",
) -> str:
    tmpl = PROMPT_TEMPLATES.get(template, PROMPT_TEMPLATES["alpaca"])
    return tmpl.format(
        instruction=row.get(instruction_col, row.get("instruction", "")),
        output=row.get(output_col, row.get("output", "")),
    )


def _validate_columns(available: list[str], instruction_col: str, output_col: str) -> None:
    """Raise ValueError if the required columns are missing from ``available``."""
    if instruction_col not in available:
        raise ValueError(
            f"Instruction column '{instruction_col}' not found. Available columns: {available}"
        )
    if output_col not in available:
        raise ValueError(f"Output column '{output_col}' not found. Available columns: {available}")


def _load_raw(
    file_path: str,
    hub_dataset_id: str = "",
    hub_split: str = "train",
    instruction_col: str = "instruction",
    output_col: str = "output",
) -> Dataset:
    """Load and validate the raw dataset, normalising columns to
    ``instruction`` / ``output``. Shared by tokenization and reference eval.

    Column presence is validated as early as possible — on the raw pandas/dict
    for local files — so an invalid request fails fast before a (potentially
    expensive) HF Dataset is constructed.
    """
    if hub_dataset_id:
        raw = load_dataset(hub_dataset_id, split=hub_split, trust_remote_code=False)
        _validate_columns(raw.column_names, instruction_col, output_col)
    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
        _validate_columns(list(df.columns), instruction_col, output_col)
        raw = Dataset.from_pandas(df)
    elif file_path.endswith(".json") and not file_path.endswith(".jsonl"):
        import json

        with open(file_path) as f:
            data = json.load(f)
        records = data if isinstance(data, list) else [data]
        available = list(records[0].keys()) if records else []
        _validate_columns(available, instruction_col, output_col)
        raw = Dataset.from_list(records)
    else:
        raw = load_dataset("json", data_files=file_path, split="train")
        _validate_columns(raw.column_names, instruction_col, output_col)

    # Normalise column names so format_prompt always sees "instruction" / "output"
    if instruction_col != "instruction":
        raw = raw.rename_column(instruction_col, "instruction")
    if output_col != "output" and output_col in raw.column_names:
        raw = raw.rename_column(output_col, "output")
    return raw


def load_instruction_pairs(
    file_path: str,
    hub_dataset_id: str = "",
    hub_split: str = "train",
    instruction_col: str = "instruction",
    output_col: str = "output",
) -> Dataset:
    """Return the raw dataset with ``instruction``/``output`` text columns, for
    generating predictions and computing reference metrics (ROUGE/BLEU)."""
    return _load_raw(
        file_path,
        hub_dataset_id=hub_dataset_id,
        hub_split=hub_split,
        instruction_col=instruction_col,
        output_col=output_col,
    )


def detect_dataset_type(columns: list[str]) -> str:
    """Classify a dataset by its columns: ``preference`` (DPO) when
    prompt/chosen/rejected are all present, else ``instruction`` (SFT)."""
    cols = {c.lower() for c in columns}
    if {"prompt", "chosen", "rejected"} <= cols:
        return "preference"
    return "instruction"


def load_preference_pairs(
    file_path: str,
    hub_dataset_id: str = "",
    hub_split: str = "train",
    prompt_col: str = "prompt",
    chosen_col: str = "chosen",
    rejected_col: str = "rejected",
) -> Dataset:
    """Load a preference dataset for DPO, normalised to ``prompt``/``chosen``/
    ``rejected`` text columns (the shape ``trl.DPOTrainer`` expects)."""
    required = (prompt_col, chosen_col, rejected_col)

    def _check(available: list[str]) -> None:
        missing = [c for c in required if c not in available]
        if missing:
            raise ValueError(
                f"Preference dataset missing column(s) {missing}. Available columns: {available}"
            )

    # Validate columns on the raw source before constructing a HF Dataset, so an
    # invalid request fails fast (and without triggering dataset fingerprinting).
    if hub_dataset_id:
        raw = load_dataset(hub_dataset_id, split=hub_split, trust_remote_code=False)
        _check(raw.column_names)
    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
        _check(list(df.columns))
        raw = Dataset.from_pandas(df)
    elif file_path.endswith(".json") and not file_path.endswith(".jsonl"):
        import json

        with open(file_path) as f:
            data = json.load(f)
        records = data if isinstance(data, list) else [data]
        _check(list(records[0].keys()) if records else [])
        raw = Dataset.from_list(records)
    else:
        raw = load_dataset("json", data_files=file_path, split="train")
        _check(raw.column_names)

    for src, dst in ((prompt_col, "prompt"), (chosen_col, "chosen"), (rejected_col, "rejected")):
        if src != dst and src in raw.column_names:
            raw = raw.rename_column(src, dst)

    drop = [c for c in raw.column_names if c not in ("prompt", "chosen", "rejected")]
    if drop:
        raw = raw.remove_columns(drop)
    return raw


def load_raw_text(
    file_path: str,
    hub_dataset_id: str = "",
    hub_split: str = "train",
    instruction_col: str = "instruction",
    output_col: str = "output",
    template: str = "alpaca",
) -> Dataset:
    """Return the dataset with a single formatted ``text`` column (untokenized).

    Used for SFTTrainer sample packing, which does its own tokenization and
    concatenates examples up to the sequence length for higher GPU efficiency.
    """
    raw = _load_raw(
        file_path,
        hub_dataset_id=hub_dataset_id,
        hub_split=hub_split,
        instruction_col=instruction_col,
        output_col=output_col,
    )
    return raw.map(
        lambda x: {"text": format_prompt(x, template=template)},
        remove_columns=raw.column_names,
    )


def load_raw_dataset(
    file_path: str,
    hub_dataset_id: str = "",
    hub_split: str = "train",
    instruction_col: str = "instruction",
    output_col: str = "output",
    template: str = "alpaca",
) -> Dataset:
    """Return the raw dataset with instruction/output columns (not yet tokenized).

    Used by finetune.py to split before tokenizing (#23) so we don't waste
    compute tokenizing examples that end up in the eval set.
    """
    return _load_raw(
        file_path,
        hub_dataset_id=hub_dataset_id,
        hub_split=hub_split,
        instruction_col=instruction_col,
        output_col=output_col,
    )


def load_multimodal(
    file_path: str,
    processor,
    max_seq_length: int = 512,
    hub_dataset_id: str = "",
    hub_split: str = "train",
    instruction_col: str = "instruction",
    output_col: str = "output",
    image_col: str = "image",
) -> "Dataset":
    """Load an image-text dataset and prepare it for VLM fine-tuning.

    Returns a dataset with ``input_ids``, ``pixel_values``, and ``labels``
    columns — the shape expected by ``transformers`` VLM trainers.

    ``processor`` should be an ``AutoProcessor`` instance for the target VLM
    (e.g. LLaVA, Qwen2-VL). Text is formatted via the alpaca template; images
    are processed with ``processor.image_processor``.
    """
    if hub_dataset_id:
        raw = load_dataset(hub_dataset_id, split=hub_split, trust_remote_code=False)
    elif file_path.endswith(".csv"):
        raw = Dataset.from_pandas(pd.read_csv(file_path))
    else:
        raw = load_dataset("json", data_files=file_path, split="train")

    template = PROMPT_TEMPLATES["alpaca"]

    def _process(batch):
        texts = [
            template.format(
                instruction=batch[instruction_col][i],
                output=batch[output_col][i],
            )
            for i in range(len(batch[instruction_col]))
        ]
        images = batch.get(image_col, [None] * len(texts))
        enc = processor(
            text=texts,
            images=images if any(img is not None for img in images) else None,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
        )
        return {
            "input_ids": enc["input_ids"].tolist(),
            "pixel_values": enc["pixel_values"].tolist()
            if "pixel_values" in enc
            else [None] * len(texts),
            "labels": enc["input_ids"].tolist(),
        }

    return raw.map(_process, batched=True, remove_columns=raw.column_names)


def load_and_tokenize(
    file_path: str,
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int = 512,
    hub_dataset_id: str = "",
    hub_split: str = "train",
    instruction_col: str = "instruction",
    output_col: str = "output",
    template: str = "alpaca",
    preloaded: Dataset | None = None,
) -> Dataset:
    """Load, format, and tokenize a dataset.

    If ``preloaded`` is given (a pre-split raw Dataset) the load step is
    skipped, avoiding redundant I/O when split-before-tokenize is active.
    """
    raw = (
        preloaded
        if preloaded is not None
        else _load_raw(
            file_path,
            hub_dataset_id=hub_dataset_id,
            hub_split=hub_split,
            instruction_col=instruction_col,
            output_col=output_col,
        )
    )

    raw = raw.map(
        lambda x: {
            "text": format_prompt(x, template=template),
            "prefix": _format_prompt_prefix(x, template=template),
        }
    )

    def _tokenize_and_mask(examples):
        full_enc = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        )
        prefix_enc = tokenizer(
            examples["prefix"],
            truncation=True,
            max_length=max_seq_length,
            add_special_tokens=False,
        )
        labels = []
        for input_ids, prefix_ids in zip(
            full_enc["input_ids"], prefix_enc["input_ids"], strict=False
        ):
            lbl = list(input_ids)
            # full_enc uses add_special_tokens=True (may prepend BOS); prefix_enc
            # does not, so prefix_ids is shorter by exactly 1 when BOS is present.
            # Shift the mask start past BOS so it stays in the loss, and the last
            # instruction token is not accidentally included.
            bos_offset = (
                1
                if tokenizer.bos_token_id is not None
                and len(input_ids) > 0
                and input_ids[0] == tokenizer.bos_token_id
                else 0
            )
            prompt_len = min(len(prefix_ids), len(lbl) - bos_offset)
            lbl[bos_offset : bos_offset + prompt_len] = [-100] * prompt_len
            labels.append(lbl)
        full_enc["labels"] = labels
        return full_enc

    tokenized = raw.map(
        _tokenize_and_mask,
        batched=True,
        remove_columns=raw.column_names,
    )
    return tokenized
