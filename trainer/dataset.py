import pandas as pd
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizer

PROMPT_TEMPLATE = """### Instruction:
{instruction}

### Response:
{output}"""


def format_prompt(row: dict, instruction_col: str = "instruction", output_col: str = "output") -> str:
    return PROMPT_TEMPLATE.format(
        instruction=row.get(instruction_col, row.get("instruction", "")),
        output=row.get(output_col, row.get("output", "")),
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
    if hub_dataset_id:
        raw = load_dataset(hub_dataset_id, split=hub_split, trust_remote_code=True)
    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
        raw = Dataset.from_pandas(df)
    elif file_path.endswith(".json") and not file_path.endswith(".jsonl"):
        import json
        with open(file_path) as f:
            data = json.load(f)
        raw = Dataset.from_list(data if isinstance(data, list) else [data])
    else:
        raw = load_dataset("json", data_files=file_path, split="train")

    # Normalise column names so format_prompt always sees "instruction" / "output"
    if instruction_col != "instruction" and instruction_col in raw.column_names:
        raw = raw.rename_column(instruction_col, "instruction")
    if output_col != "output" and output_col in raw.column_names:
        raw = raw.rename_column(output_col, "output")

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
