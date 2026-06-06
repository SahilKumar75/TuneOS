import pandas as pd
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizer

# Prompt templates keyed by name. Each must contain {instruction} and {output}.
PROMPT_TEMPLATES: dict[str, str] = {
    "alpaca": "### Instruction:\n{instruction}\n\n### Response:\n{output}",
    "chatml": (
        "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
    ),
    "llama3": (
        "<|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n{output}<|eot_id|>"
    ),
    "phi3": "<|user|>\n{instruction}<|end|>\n<|assistant|>\n{output}<|end|>",
    "zephyr": "<|user|>\n{instruction}</s>\n<|assistant|>\n{output}</s>",
}
# Back-compat: callers importing PROMPT_TEMPLATE get the default (alpaca).
PROMPT_TEMPLATE = PROMPT_TEMPLATES["alpaca"]


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
    if hub_dataset_id:
        raw = load_dataset(hub_dataset_id, split=hub_split, trust_remote_code=False)
    elif file_path.endswith(".csv"):
        raw = Dataset.from_pandas(pd.read_csv(file_path))
    elif file_path.endswith(".json") and not file_path.endswith(".jsonl"):
        import json

        with open(file_path) as f:
            data = json.load(f)
        raw = Dataset.from_list(data if isinstance(data, list) else [data])
    else:
        raw = load_dataset("json", data_files=file_path, split="train")

    missing = [c for c in (prompt_col, chosen_col, rejected_col) if c not in raw.column_names]
    if missing:
        raise ValueError(
            f"Preference dataset missing column(s) {missing}. Available columns: {raw.column_names}"
        )

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


def load_and_tokenize(
    file_path: str,
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int = 512,
    hub_dataset_id: str = "",
    hub_split: str = "train",
    instruction_col: str = "instruction",
    output_col: str = "output",
    template: str = "alpaca",
) -> Dataset:
    """
    Load from a local file or HF Hub dataset, apply column mapping,
    format as instruction prompts, and tokenize.
    """
    raw = _load_raw(
        file_path,
        hub_dataset_id=hub_dataset_id,
        hub_split=hub_split,
        instruction_col=instruction_col,
        output_col=output_col,
    )

    raw = raw.map(lambda x: {"text": format_prompt(x, template=template)})
    tokenized = raw.map(
        lambda x: tokenizer(
            x["text"],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        ),
        batched=True,
        remove_columns=raw.column_names,
    )
    tokenized = tokenized.map(lambda x: {"labels": x["input_ids"].copy()})
    return tokenized
