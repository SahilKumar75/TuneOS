"""Prompt/chat templates and model→template auto-selection.

Deliberately dependency-free (stdlib only) so both the Reflex app and the
framework-heavy trainer can import it without dragging in transformers/torch.
This is the single source of truth — `trainer.dataset` and the app's wizard
state both import from here.
"""

from __future__ import annotations

# Each template must contain the {instruction} and {output} placeholders.
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
    "gemma": (
        "<start_of_turn>user\n{instruction}<end_of_turn>\n"
        "<start_of_turn>model\n{output}<end_of_turn>"
    ),
    "mistral": "[INST] {instruction} [/INST] {output}",
}

# Back-compat: callers importing PROMPT_TEMPLATE get the default (alpaca).
PROMPT_TEMPLATE = PROMPT_TEMPLATES["alpaca"]

_INSTRUCT_HINTS = ("instruct", "-it", "chat", "sft", "zephyr", "hermes", "dolphin", "openchat")
_INSTRUCT_TAGS = {"instruction-tuned", "conversational", "chat"}


def auto_prompt_template_for(
    model_type: str = "",
    model_id: str = "",
    tags: list[str] | None = None,
) -> str:
    """Pick the prompt template that matches a model's native chat format.

    Maps by model family (from HF config ``model_type`` and the repo id). Falls
    back to ``chatml`` for unknown instruct/chat models and ``alpaca`` for raw
    base models. Never raises; returns a key guaranteed to be in PROMPT_TEMPLATES.
    """
    mt = (model_type or "").lower()
    mid = (model_id or "").lower()
    tagset = {t.lower() for t in (tags or [])}
    is_instruct = any(h in mid for h in _INSTRUCT_HINTS) or bool(tagset & _INSTRUCT_TAGS)

    # Family-native formats.
    if "phi3" in mt or "phi-3" in mid or "phi3" in mid:
        return "phi3"
    if "gemma" in mt or "gemma" in mid:
        return "gemma"
    if "qwen" in mt or "qwen" in mid:
        return "chatml"  # Qwen ships with a ChatML chat template
    if "llama" in mt or "llama" in mid:
        if any(k in mid for k in ("llama-3", "llama3", "meta-llama-3")):
            return "llama3"
        return "chatml" if is_instruct else "alpaca"  # llama-2 base → alpaca
    if "mixtral" in mt or "mistral" in mt or "mistral" in mid or "mixtral" in mid:
        return "mistral" if is_instruct else "alpaca"

    return "chatml" if is_instruct else "alpaca"
