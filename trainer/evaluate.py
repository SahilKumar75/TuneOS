import math

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer


def evaluate_model(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, test_dataset) -> dict:
    """
    Compute perplexity on a held-out dataset sample.
    Returns {"perplexity": float, "bleu": None}.
    BLEU is omitted — instruction-following datasets have no reference outputs.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    loader = DataLoader(test_dataset, batch_size=1)

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(model.device)
            labels = batch["labels"].to(model.device)
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            if loss is None or torch.isnan(loss):
                continue
            n_tokens = (labels != -100).sum().item()
            if n_tokens == 0:
                continue
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

    if total_tokens == 0:
        return {"perplexity": None, "bleu": None}

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(min(avg_loss, 20))  # cap at e^20 to avoid inf
    return {"perplexity": round(perplexity, 3), "bleu": None}
