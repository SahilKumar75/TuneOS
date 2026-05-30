from transformers import PreTrainedModel, PreTrainedTokenizer


def evaluate_model(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, test_dataset):
    """
    Evaluate model using Perplexity and BLEU metrics.
    """
    model.eval()

    # Metrics are loaded lazily inside the (not-yet-implemented) evaluation
    # loop. Perplexity and BLEU are the planned metrics for this function.
    # The concrete implementation depends on the target dataset format.
    print("Evaluating model...")

    return {"perplexity": None, "bleu": None}
