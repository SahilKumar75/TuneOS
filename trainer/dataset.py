import pandas as pd
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizer

PROMPT_TEMPLATE = """### Instruction:
{instruction}

### Response:
{output}"""


def format_prompt(
    row: dict, instruction_col: str = "instruction", output_col: str = "output"
) -> str:
    return PROMPT_TEMPLATE.format(
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


def load_and_tokenize(
    file_path: str,
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int = 512,
    hub_dataset_id: str = "",
    hub_split: str = "train",
    instruction_col: str = "instruction",
    output_col: str = "output",
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

    raw = raw.map(lambda x: {"text": format_prompt(x)})
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
