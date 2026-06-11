"""Fine-tune wizard — Step 7: Deploy."""

from __future__ import annotations

import reflex as rx

from app.components.finetune.shared import _card, _label, _section_heading
from app.state.deploy_state import DeployState
from app.state.finetune_state import FinetuneState
from app.styles import c

_GGUF_QUANTS = ["Q4_K_M", "Q5_K_M", "Q8_0", "F16"]


def _deploy_target_row(
    target_key: str,
    label: str,
    description: str,
    icon: str,
    is_checked,
) -> rx.Component:
    return rx.hstack(
        rx.checkbox(
            checked=is_checked,
            on_change=lambda _: DeployState.toggle_deploy_target(target_key),
            size="2",
        ),
        rx.vstack(
            rx.text(label, font_weight="500", font_size="0.88rem", color=c("text_primary")),
            rx.text(description, font_size="0.76rem", color=c("text_muted")),
            spacing="0",
        ),
        spacing="3",
        align="start",
        padding="10px 0",
        border_bottom="1px solid",
        border_color=c("border"),
        width="100%",
    )


def _deploy_body() -> rx.Component:
    """Shared deploy controls used by both linear and workspace layouts."""
    return rx.vstack(
        rx.text(
            "Choose how you want to export or share your fine-tuned adapter.",
            font_size="0.86rem",
            color=c("text_secondary"),
            margin_bottom="16px",
        ),
        # Target selector
        _card(
            rx.vstack(
                _deploy_target_row(
                    "adapter",
                    "Download adapter",
                    "Zip the LoRA adapter files (~100 MB) — works with PEFT/Transformers",
                    "download",
                    DeployState.deploy_adapter,
                ),
                _deploy_target_row(
                    "merged",
                    "Download merged model",
                    "Merge adapter into base model → full standalone safetensors (~14 GB for 7B)",
                    "layers",
                    DeployState.deploy_merged,
                ),
                _deploy_target_row(
                    "hub",
                    "Push to Hugging Face Hub",
                    "Upload adapter to a public or private HF repository",
                    "globe",
                    DeployState.deploy_hub,
                ),
                _deploy_target_row(
                    "gguf",
                    "Export as GGUF",
                    "Convert to GGUF format for use with Ollama or llama.cpp (CPU inference)",
                    "cpu",
                    DeployState.deploy_gguf,
                ),
                _deploy_target_row(
                    "github",
                    "Push to GitHub",
                    "Commit adapter files to a GitHub repository using Git LFS",
                    "github",
                    DeployState.deploy_github,
                ),
                spacing="0",
            )
        ),
        # HF Hub fields
        rx.cond(
            DeployState.deploy_hub,
            _card(
                rx.vstack(
                    rx.text(
                        "Hugging Face Hub",
                        font_weight="600",
                        font_size="0.88rem",
                        color=c("text_primary"),
                    ),
                    rx.grid(
                        rx.vstack(
                            _label("HF Token"),
                            rx.input(
                                type="password",
                                placeholder="hf_xxxxxxxxxxxxx",
                                value=DeployState.hf_token_input,
                                on_change=DeployState.set_hf_token_input,
                                width="100%",
                            ),
                            spacing="1",
                        ),
                        rx.vstack(
                            _label("Repository name (e.g. myuser/my-chatbot-lora)"),
                            rx.input(
                                placeholder="username/repo-name",
                                value=DeployState.hf_repo_name,
                                on_change=DeployState.set_hf_repo_name,
                                width="100%",
                            ),
                            spacing="1",
                        ),
                        columns="2",
                        spacing="3",
                        width="100%",
                    ),
                    # Version overwrite warning
                    rx.cond(
                        DeployState.hf_repo_name != "",
                        rx.callout(
                            rx.hstack(
                                rx.icon("alert-triangle", size=14),
                                rx.text(
                                    "Pushing to an existing repo overwrites it. "
                                    "Consider versioning: append -v2 or -v3 to the repo name."
                                ),
                                spacing="2",
                                align="start",
                            ),
                            color_scheme="amber",
                            size="1",
                        ),
                        rx.fragment(),
                    ),
                    rx.button(
                        rx.cond(
                            DeployState.push_status == "pushing",
                            rx.hstack(rx.spinner(size="2"), rx.text("Pushing..."), spacing="2"),
                            rx.text("Push to Hub"),
                        ),
                        on_click=DeployState.push_to_hub,
                        disabled=DeployState.push_status == "pushing",
                        color_scheme="blue",
                        size="2",
                    ),
                    rx.cond(
                        DeployState.push_status == "done",
                        rx.callout(
                            rx.hstack(
                                rx.icon("circle-check", size=14),
                                rx.text(f"Pushed to {DeployState.push_repo_url}"),
                                spacing="2",
                            ),
                            color_scheme="green",
                            size="1",
                        ),
                        rx.fragment(),
                    ),
                    rx.cond(
                        DeployState.push_error != "",
                        rx.callout(DeployState.push_error, color_scheme="red", size="1"),
                        rx.fragment(),
                    ),
                    spacing="3",
                )
            ),
            rx.fragment(),
        ),
        # GGUF fields
        rx.cond(
            DeployState.deploy_gguf,
            _card(
                rx.vstack(
                    rx.text(
                        "GGUF Export",
                        font_weight="600",
                        font_size="0.88rem",
                        color=c("text_primary"),
                    ),
                    rx.callout(
                        "GGUF export requires the model to be merged first. "
                        "Enable 'Download merged model' above to trigger the merge step.",
                        color_scheme="amber",
                        size="1",
                    ),
                    rx.hstack(
                        rx.vstack(
                            _label("Quantization"),
                            rx.select.root(
                                rx.select.trigger(width="160px"),
                                rx.select.content(
                                    *[rx.select.item(q, value=q) for q in _GGUF_QUANTS],
                                ),
                                value=DeployState.gguf_quantization,
                                on_change=DeployState.set_gguf_quantization,
                            ),
                            spacing="1",
                        ),
                        rx.button(
                            rx.cond(
                                DeployState.gguf_status == "exporting",
                                rx.hstack(
                                    rx.spinner(size="2"),
                                    rx.text("Exporting..."),
                                    spacing="2",
                                ),
                                rx.text("Export GGUF"),
                            ),
                            on_click=DeployState.start_gguf_export,
                            disabled=DeployState.gguf_status == "exporting",
                            color_scheme="blue",
                            size="2",
                            align_self="flex-end",
                        ),
                        spacing="3",
                        align="end",
                    ),
                    spacing="3",
                )
            ),
            rx.fragment(),
        ),
        # GitHub fields
        rx.cond(
            DeployState.deploy_github,
            _card(
                rx.vstack(
                    rx.text(
                        "GitHub Push",
                        font_weight="600",
                        font_size="0.88rem",
                        color=c("text_primary"),
                    ),
                    rx.grid(
                        rx.vstack(
                            _label("Repository URL (HTTPS)"),
                            rx.input(
                                placeholder="https://github.com/user/repo",
                                value=DeployState.github_repo_url,
                                on_change=DeployState.set_github_repo_url,
                                width="100%",
                            ),
                            spacing="1",
                        ),
                        rx.vstack(
                            _label("GitHub Token (needs repo scope)"),
                            rx.input(
                                type="password",
                                placeholder="ghp_xxxxxxxxxxxxx",
                                value=DeployState.github_token,
                                on_change=DeployState.set_github_token,
                                width="100%",
                            ),
                            spacing="1",
                        ),
                        columns="2",
                        spacing="3",
                        width="100%",
                    ),
                    rx.button(
                        rx.cond(
                            DeployState.github_push_status == "pushing",
                            rx.hstack(rx.spinner(size="2"), rx.text("Pushing..."), spacing="2"),
                            rx.text("Push to GitHub"),
                        ),
                        on_click=DeployState.push_to_github,
                        disabled=DeployState.github_push_status == "pushing",
                        color_scheme="blue",
                        size="2",
                    ),
                    rx.cond(
                        DeployState.github_push_status == "done",
                        rx.callout(
                            f"Pushed to {DeployState.github_repo_url}",
                            color_scheme="green",
                            size="1",
                        ),
                        rx.fragment(),
                    ),
                    spacing="3",
                )
            ),
            rx.fragment(),
        ),
        # Quick actions
        rx.hstack(
            rx.button(
                rx.hstack(rx.icon("download", size=14), rx.text("Download adapter"), spacing="2"),
                on_click=DeployState.download_adapter,
                color_scheme="blue",
                variant="soft",
                size="2",
            ),
            rx.cond(
                DeployState.deploy_merged,
                rx.button(
                    rx.cond(
                        DeployState.merge_status == "merging",
                        rx.hstack(rx.spinner(size="2"), rx.text("Merging..."), spacing="2"),
                        rx.hstack(
                            rx.icon("layers", size=14),
                            rx.text("Merge & download"),
                            spacing="2",
                        ),
                    ),
                    on_click=DeployState.start_merge,
                    disabled=DeployState.merge_status == "merging",
                    color_scheme="blue",
                    variant="soft",
                    size="2",
                ),
                rx.fragment(),
            ),
            spacing="3",
            wrap="wrap",
        ),
        # Deploy log
        rx.cond(
            DeployState.deploy_log != "",
            _card(
                rx.vstack(
                    rx.text(
                        "Activity log",
                        font_size="0.78rem",
                        font_weight="600",
                        color=c("text_secondary"),
                    ),
                    rx.box(
                        rx.text(
                            DeployState.deploy_log,
                            font_size="0.76rem",
                            color=c("text_secondary"),
                            font_family="monospace",
                            white_space="pre-wrap",
                        ),
                        background=c("bg_input"),
                        border_radius="8px",
                        padding="12px",
                        max_height="200px",
                        overflow_y="auto",
                        width="100%",
                    ),
                    spacing="2",
                )
            ),
            rx.fragment(),
        ),
        rx.box(height="8px"),
        rx.button(
            "Start a new fine-tune →",
            on_click=rx.redirect("/finetune"),
            color_scheme="gray",
            variant="soft",
            size="2",
        ),
        spacing="4",
        width="100%",
        align_items="flex-start",
    )


def step7_deploy() -> rx.Component:
    """Linear wizard layout for Step 7 (deploy)."""
    return rx.vstack(
        _section_heading("Deploy your model"),
        _deploy_body(),
        spacing="4",
        width="100%",
        align_items="flex-start",
    )


def step7_workspace_panel() -> rx.Component:
    """Full-screen workspace panel for Step 7."""
    return rx.box(
        rx.vstack(
            rx.button(
                rx.hstack(
                    rx.icon("arrow-left", size=13),
                    rx.text("Back to workspace"),
                    spacing="1",
                ),
                on_click=FinetuneState.go_to_step(6),
                variant="ghost",
                color_scheme="gray",
                size="1",
                align_self="flex-start",
                margin_bottom="4px",
            ),
            rx.cond(
                DeployState.training_status != "done",
                rx.callout(
                    "Training must complete before you can deploy. "
                    "Start training from the Configure panel.",
                    icon="info",
                    color_scheme="amber",
                    size="2",
                    width="100%",
                ),
                _deploy_body(),
            ),
            spacing="0",
            width="100%",
            align_items="flex-start",
        ),
        padding="20px",
        width="100%",
        overflow_y="auto",
        height="100%",
    )
