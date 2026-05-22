from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizer
from trainer.config import TrainingConfig, ModelConfig
import pandas as pd

PROMPT_TEMPLATE = """### Instruction:
{instruction}

### Response:
{output}"""

def format_prompt(row: dict) -> str:
    return PROMPT_TEMPLATE.format(
        instruction=row.get("instruction", ""),
        output=row.get("output", ""),
    )

def load_and_tokenize(
    file_path: str,
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int = 512,
) -> Dataset:
    """
    Load JSONL or CSV, format as instruction prompts, tokenize.
    """
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
        raw = Dataset.from_pandas(df)
    else:
        raw = load_dataset("json", data_files=file_path, split="train")

    def tokenize(batch):
        prompts = [format_prompt(row) for row in batch]  # adjust if batched
        return tokenizer(
            prompts,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        )

    # Map with formatted prompts first
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
