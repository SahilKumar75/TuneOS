"""TuneOS — Fine-tuning wizard (/finetune) — 7-step flow."""

from __future__ import annotations

import reflex as rx

from app.components.finetune.step1_model import _step1
from app.components.finetune.step2_intent import _step2
from app.components.finetune.step3_data import _step3
from app.components.finetune.step5_train import step5_panel
from app.components.finetune.step6_results import step6_panel
from app.components.finetune.step7_deploy import step7_workspace_panel
from app.state.experiment_state import ExperimentState, ModelRegistryState
from app.state.finetune_state import FinetuneState
from app.state.training_poller_state import TrainingPollerState
from app.styles import c

_STEP_LABELS = ["Model", "Intent", "Data", "Configure", "Train", "Results", "Deploy"]

_LR_PRESETS = [
    ("1e-4", "Slow & careful"),
    ("2e-4", "Balanced (recommended)"),
    ("5e-4", "Fast learning"),
]


# ── Shared helpers ────────────────────────────────────────────────
def _card(
    *children,
    padding: str = "20px",
    width: str = "100%",
    background: str | None = None,
    **props,
) -> rx.Component:
    return rx.box(
        *children,
        background=c("bg_card") if background is None else background,
        border="1px solid",
        border_color=c("border"),
        border_radius="12px",
        padding=padding,
        width=width,
        **props,
    )


def _label(text: str) -> rx.Component:
    return rx.text(
        text, font_size="0.8rem", font_weight="500", color=c("text_secondary"), margin_bottom="6px"
    )


def _section_heading(text: str) -> rx.Component:
    return rx.text(
        text, font_size="1.05rem", font_weight="600", color=c("text_primary"), margin_bottom="16px"
    )


def _nav_buttons(
    back_label: str = "← Back",
    next_label: str = "Next →",
    next_disabled: bool = False,
    next_event=None,
    show_back: bool = True,
) -> rx.Component:
    return rx.hstack(
        rx.button(
            back_label,
            on_click=FinetuneState.prev_step,
            variant="soft",
            color_scheme="gray",
            size="2",
        )
        if show_back
        else rx.fragment(),
        rx.spacer(),
        rx.button(
            next_label,
            on_click=next_event or FinetuneState.next_step,
            disabled=next_disabled,
            size="3",
            color_scheme="blue",
        ),
        width="100%",
        padding_top="16px",
    )


def _badge_status(status: str) -> rx.Component:
    color = rx.match(
        status,
        ("running", "blue"),
        ("done", "green"),
        ("failed", "red"),
        "gray",
    )
    return rx.badge(status.upper(), color_scheme=color, size="2")


# ── Progress bar ──────────────────────────────────────────────────
def _step_dot(index: int) -> rx.Component:
    step_num = index + 1
    is_done = FinetuneState.current_step > step_num
    is_active = FinetuneState.current_step == step_num
    return rx.vstack(
        rx.box(
            rx.cond(
                is_done,
                rx.icon("check", size=12, color="white"),
                rx.text(
                    str(step_num),
                    font_size="0.72rem",
                    font_weight="600",
                    color=rx.cond(is_active, "white", c("text_muted")),
                ),
            ),
            width="26px",
            height="26px",
            border_radius="50%",
            background=rx.cond(
                is_done, c("success"), rx.cond(is_active, c("accent"), c("bg_input"))
            ),
            border="2px solid",
            border_color=rx.cond(is_active | is_done, c("accent"), c("border")),
            display="flex",
            align_items="center",
            justify_content="center",
        ),
        rx.text(
            _STEP_LABELS[index],
            font_size="0.68rem",
            color=rx.cond(is_active, c("text_primary"), c("text_muted")),
            font_weight=rx.cond(is_active, "500", "400"),
        ),
        spacing="1",
        align="center",
    )


def _progress_bar() -> rx.Component:
    return rx.hstack(
        *[
            rx.hstack(
                _step_dot(i),
                rx.box(
                    height="2px",
                    flex="1",
                    background=rx.cond(
                        FinetuneState.current_step > i + 1, c("accent"), c("border")
                    ),
                    min_width="20px",
                )
                if i < len(_STEP_LABELS) - 1
                else rx.fragment(),
                spacing="0",
                align="center",
                flex="1" if i < len(_STEP_LABELS) - 1 else "0",
            )
            for i in range(len(_STEP_LABELS))
        ],
        width="100%",
        align="center",
        justify="center",
        margin_bottom="32px",
    )


# ── CSS keyframes & workspace animations ─────────────────────────
def _workspace_styles() -> rx.Component:
    return rx.el.style(
        """
        @keyframes workspace-enter {
            from { opacity: 0; transform: translateX(-8px); }
            to   { opacity: 1; transform: translateX(0); }
        }
        @keyframes slide-up {
            from { opacity: 0; transform: translateY(40px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes idle-pulse {
            0%, 100% { opacity: 0.5; transform: scale(1); }
            50%       { opacity: 0.15; transform: scale(1.35); }
        }
        .workspace-root { animation: workspace-enter 0.28s ease-out; }
        .step6-reveal   { animation: slide-up 0.4s ease-out; }
        .idle-pulse-ring { animation: idle-pulse 2.2s ease-in-out infinite; }
        """
    )


# ── Workspace header ──────────────────────────────────────────────
def _workspace_header() -> rx.Component:
    return rx.box(
        rx.hstack(
            rx.hstack(
                rx.icon("flask-conical", size=18, color=c("accent")),
                rx.text(
                    "TuneOS",
                    font_weight="700",
                    color=c("text_primary"),
                    font_size="0.95rem",
                ),
                rx.text("·", color=c("text_muted"), font_size="0.9rem"),
                rx.text(
                    rx.cond(
                        FinetuneState.experiment_name != "",
                        FinetuneState.experiment_name,
                        "New experiment",
                    ),
                    font_size="0.84rem",
                    color=c("text_secondary"),
                    max_width="280px",
                    overflow="hidden",
                    text_overflow="ellipsis",
                    white_space="nowrap",
                ),
                spacing="2",
                align="center",
            ),
            rx.spacer(),
            rx.button(
                rx.hstack(
                    rx.icon("cloud-upload", size=14),
                    rx.text("Deploy Model"),
                    spacing="2",
                    align="center",
                ),
                on_click=FinetuneState.go_to_step(7),
                disabled=TrainingPollerState.training_status != "done",
                color_scheme="blue",
                variant=rx.cond(TrainingPollerState.training_status == "done", "solid", "soft"),
                size="2",
                opacity=rx.cond(TrainingPollerState.training_status == "done", "1", "0.45"),
                cursor=rx.cond(
                    TrainingPollerState.training_status == "done", "pointer", "not-allowed"
                ),
            ),
            width="100%",
            align="center",
            padding="0 20px",
        ),
        background=c("bg_card"),
        border_bottom="1px solid",
        border_color=c("border"),
        height="52px",
        display="flex",
        align_items="center",
        width="100%",
        flex_shrink="0",
    )


# ── Workspace sidebar ─────────────────────────────────────────────
def _sidebar_icon(step_num: int, icon_name: str, tooltip: str, gated: bool = False) -> rx.Component:
    is_active = FinetuneState.current_step == step_num
    is_done = FinetuneState.current_step > step_num
    # Use rx.cond(gated, False, <var>) instead of <var> & ~gated to avoid
    # ~False == -1 (Python bitwise NOT on a plain bool default) corrupting
    # Reflex Var boolean expressions.
    show_check = rx.cond(gated, False, is_done)
    show_active_color = rx.cond(gated, False, is_active)
    return rx.box(
        rx.cond(
            show_check,
            rx.icon("check", size=15, color=c("success")),
            rx.icon(
                icon_name,
                size=15,
                color=rx.cond(show_active_color, "white", c("text_muted")),
            ),
        ),
        title=tooltip,
        width="36px",
        height="36px",
        border_radius="10px",
        background=rx.cond(
            gated,
            "transparent",
            rx.cond(
                is_active,
                c("accent"),
                rx.cond(is_done, c("accent_soft"), "transparent"),
            ),
        ),
        display="flex",
        align_items="center",
        justify_content="center",
        cursor=rx.cond(gated, "not-allowed", "pointer"),
        opacity=rx.cond(gated, "0.35", "1"),
        on_click=rx.cond(gated, rx.prevent_default, FinetuneState.go_to_step(step_num)),
        _hover={
            "background": rx.cond(
                gated, "transparent", rx.cond(is_active, c("accent"), c("bg_input"))
            )
        },
        transition="background 0.15s ease, opacity 0.15s ease",
    )


def _workspace_sidebar() -> rx.Component:
    training_done = TrainingPollerState.training_status == "done"
    return rx.vstack(
        _sidebar_icon(4, "settings-2", "Configure"),
        _sidebar_icon(5, "activity", "Training"),
        _sidebar_icon(6, "bar-chart-2", "Results", gated=~training_done),
        _sidebar_icon(7, "cloud-upload", "Deploy", gated=~training_done),
        spacing="2",
        align="center",
        padding="16px 6px",
        width="48px",
        min_width="48px",
        border_right="1px solid",
        border_color=c("border"),
        background=c("bg_card"),
        height="100%",
        flex_shrink="0",
    )


# ── Step 4 panel (Configure) ──────────────────────────────────────
def _step4_panel() -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.hstack(
                    rx.icon("settings-2", size=14, color=c("accent")),
                    rx.text(
                        "Configure",
                        font_size="0.82rem",
                        font_weight="600",
                        color=c("text_primary"),
                    ),
                    spacing="2",
                    align="center",
                ),
                rx.spacer(),
                rx.hstack(
                    rx.text("Simple", font_size="0.76rem", color=c("text_secondary")),
                    rx.switch(
                        checked=FinetuneState.ui_mode == "advanced",
                        on_change=FinetuneState.toggle_ui_mode,
                        size="1",
                    ),
                    rx.text("Advanced", font_size="0.76rem", color=c("text_secondary")),
                    spacing="2",
                    align="center",
                ),
                width="100%",
                align="center",
                margin_bottom="14px",
            ),
            # Simple config
            _card(
                rx.vstack(
                    rx.grid(
                        rx.vstack(
                            _label("Epochs"),
                            rx.input(
                                value=FinetuneState.epochs.to_string(),
                                on_change=FinetuneState.set_epochs,
                                type="number",
                                width="100%",
                            ),
                            rx.text(
                                "One full pass through your dataset",
                                font_size="0.72rem",
                                color=c("text_muted"),
                            ),
                            spacing="1",
                        ),
                        rx.vstack(
                            _label("Learning rate"),
                            rx.select.root(
                                rx.select.trigger(width="100%"),
                                rx.select.content(
                                    *[
                                        rx.select.item(f"{lr} — {desc}", value=lr)
                                        for lr, desc in _LR_PRESETS
                                    ]
                                ),
                                value=FinetuneState.learning_rate,
                                on_change=FinetuneState.set_learning_rate,
                            ),
                            spacing="1",
                        ),
                        rx.vstack(
                            _label("Technique"),
                            rx.text(
                                FinetuneState.technique_label,
                                font_size="0.88rem",
                                font_weight="500",
                                color=c("accent"),
                            ),
                            rx.text(
                                "Change in Step 1",
                                font_size="0.72rem",
                                color=c("text_muted"),
                            ),
                            spacing="1",
                        ),
                        columns="3",
                        spacing="4",
                        width="100%",
                    ),
                    spacing="0",
                )
            ),
            # Advanced mode
            rx.cond(
                FinetuneState.ui_mode == "advanced",
                _card(
                    rx.vstack(
                        rx.text(
                            "Advanced hyperparameters",
                            font_size="0.88rem",
                            font_weight="600",
                            color=c("text_primary"),
                            margin_bottom="12px",
                        ),
                        rx.grid(
                            rx.vstack(
                                _label("LoRA rank (r)"),
                                rx.slider(
                                    min=4,
                                    max=128,
                                    step=4,
                                    default_value=[FinetuneState.lora_r],
                                    on_value_commit=FinetuneState.set_lora_r,
                                ),
                                rx.text(
                                    FinetuneState.lora_r,
                                    font_size="0.82rem",
                                    color=c("text_secondary"),
                                ),
                                spacing="1",
                            ),
                            rx.vstack(
                                _label("LoRA alpha"),
                                rx.input(
                                    value=FinetuneState.lora_alpha.to_string(),
                                    on_change=FinetuneState.set_lora_alpha,
                                    type="number",
                                    width="100%",
                                ),
                                spacing="1",
                            ),
                            rx.vstack(
                                _label("LoRA dropout"),
                                rx.slider(
                                    min=0.0,
                                    max=0.3,
                                    step=0.01,
                                    default_value=[FinetuneState.lora_dropout],
                                    on_value_commit=FinetuneState.set_lora_dropout,
                                ),
                                rx.text(
                                    FinetuneState.lora_dropout,
                                    font_size="0.82rem",
                                    color=c("text_secondary"),
                                ),
                                spacing="1",
                            ),
                            rx.vstack(
                                _label("Batch size"),
                                rx.select.root(
                                    rx.select.trigger(width="100%"),
                                    rx.select.content(
                                        *[
                                            rx.select.item(str(v), value=str(v))
                                            for v in [1, 2, 4, 8, 16]
                                        ]
                                    ),
                                    value=FinetuneState.batch_size.to_string(),
                                    on_change=FinetuneState.set_batch_size,
                                ),
                                spacing="1",
                            ),
                            rx.vstack(
                                _label("Max seq length"),
                                rx.select.root(
                                    rx.select.trigger(width="100%"),
                                    rx.select.content(
                                        *[
                                            rx.select.item(str(v), value=str(v))
                                            for v in [128, 256, 512, 1024, 2048]
                                        ]
                                    ),
                                    value=FinetuneState.max_seq_length.to_string(),
                                    on_change=FinetuneState.set_max_seq_length,
                                ),
                                spacing="1",
                            ),
                            rx.vstack(
                                _label("Grad. accum."),
                                rx.select.root(
                                    rx.select.trigger(width="100%"),
                                    rx.select.content(
                                        *[
                                            rx.select.item(str(v), value=str(v))
                                            for v in [1, 2, 4, 8, 16]
                                        ]
                                    ),
                                    value=FinetuneState.gradient_accumulation_steps.to_string(),
                                    on_change=FinetuneState.set_gradient_accumulation_steps,
                                ),
                                spacing="1",
                            ),
                            rx.vstack(
                                _label("LR scheduler"),
                                rx.select.root(
                                    rx.select.trigger(width="100%"),
                                    rx.select.content(
                                        *[
                                            rx.select.item(v, value=v)
                                            for v in [
                                                "cosine",
                                                "linear",
                                                "constant",
                                                "cosine_with_restarts",
                                            ]
                                        ]
                                    ),
                                    value=FinetuneState.lr_scheduler,
                                    on_change=FinetuneState.set_lr_scheduler,
                                ),
                                spacing="1",
                            ),
                            rx.vstack(
                                _label("BF16"),
                                rx.switch(
                                    checked=FinetuneState.bf16,
                                    on_change=FinetuneState.set_bf16,
                                    size="2",
                                ),
                                rx.text(
                                    "A100/H100 only",
                                    font_size="0.72rem",
                                    color=c("text_muted"),
                                ),
                                spacing="1",
                            ),
                            rx.vstack(
                                _label("Experiment name"),
                                rx.input(
                                    placeholder="my-run-1",
                                    value=FinetuneState.experiment_name,
                                    on_change=FinetuneState.set_experiment_name,
                                    width="100%",
                                ),
                                spacing="1",
                            ),
                            columns="3",
                            spacing="4",
                            width="100%",
                        ),
                        spacing="0",
                    )
                ),
                rx.fragment(),
            ),
            # Run summary
            _card(
                rx.vstack(
                    rx.text(
                        "Run summary",
                        font_size="0.82rem",
                        font_weight="600",
                        color=c("text_secondary"),
                        margin_bottom="8px",
                    ),
                    rx.grid(
                        rx.vstack(
                            rx.text("Model", font_size="0.72rem", color=c("text_muted")),
                            rx.text(
                                FinetuneState.effective_model_name,
                                font_size="0.84rem",
                                font_weight="500",
                                color=c("text_primary"),
                            ),
                            spacing="0",
                        ),
                        rx.vstack(
                            rx.text("Dataset", font_size="0.72rem", color=c("text_muted")),
                            rx.text(
                                FinetuneState.dataset_name,
                                font_size="0.84rem",
                                font_weight="500",
                                color=c("text_primary"),
                            ),
                            spacing="0",
                        ),
                        rx.vstack(
                            rx.text("Technique", font_size="0.72rem", color=c("text_muted")),
                            rx.text(
                                FinetuneState.technique_label,
                                font_size="0.84rem",
                                font_weight="500",
                                color=c("text_primary"),
                            ),
                            spacing="0",
                        ),
                        rx.vstack(
                            rx.text("Training", font_size="0.72rem", color=c("text_muted")),
                            rx.text(
                                FinetuneState.epochs.to_string()
                                + " epochs · lr="
                                + FinetuneState.learning_rate,
                                font_size="0.82rem",
                                font_weight="500",
                                color=c("text_primary"),
                            ),
                            spacing="0",
                        ),
                        columns="2",
                        spacing="4",
                        width="100%",
                    ),
                    spacing="0",
                ),
                background=c("bg_input"),
            ),
            # Start Training button (no back button in workspace)
            rx.hstack(
                rx.spacer(),
                rx.button(
                    rx.cond(
                        TrainingPollerState.is_starting,
                        rx.hstack(rx.spinner(size="2"), rx.text("Starting…"), spacing="2"),
                        rx.hstack(
                            rx.icon("zap", size=14),
                            rx.text("Start Training →"),
                            spacing="2",
                        ),
                    ),
                    on_click=TrainingPollerState.start_training,
                    disabled=~FinetuneState.can_start_training
                    | TrainingPollerState.is_starting
                    | (TrainingPollerState.training_status == "running"),
                    size="3",
                    color_scheme="blue",
                ),
                width="100%",
                padding_top="4px",
            ),
            spacing="4",
            width="100%",
            align_items="flex-start",
        ),
        padding="20px",
        width="50%",
        border_right="1px solid",
        border_color=c("border"),
        overflow_y="auto",
        height="100%",
        flex_shrink="0",
    )


# ── Workspace layout (steps 4-7) ──────────────────────────────────
def _workspace_layout() -> rx.Component:
    return rx.vstack(
        _workspace_styles(),
        _workspace_header(),
        rx.hstack(
            _workspace_sidebar(),
            rx.cond(
                FinetuneState.current_step == 7,
                step7_workspace_panel(),
                rx.vstack(
                    rx.hstack(
                        _step4_panel(),
                        step5_panel(),
                        width="100%",
                        align_items="stretch",
                        spacing="0",
                        flex="1",
                        min_height="0",
                        overflow="hidden",
                    ),
                    rx.cond(
                        TrainingPollerState.training_status == "done",
                        step6_panel(),
                        rx.fragment(),
                    ),
                    width="100%",
                    height="100%",
                    spacing="0",
                    align_items="flex-start",
                    overflow="auto",
                ),
            ),
            width="100%",
            align_items="stretch",
            flex="1",
            spacing="0",
            min_height="0",
            overflow="hidden",
        ),
        width="100%",
        height="100vh",
        spacing="0",
        class_name="workspace-root",
        overflow="hidden",
    )


# ── Wizard layout (steps 1-3) ─────────────────────────────────────
def _wizard_layout() -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.icon("flask-conical", size=20, color=c("accent")),
                rx.text(
                    "Fine-tune a model",
                    font_size="1.25rem",
                    font_weight="700",
                    color=c("text_primary"),
                ),
                spacing="2",
                align="center",
                margin_bottom="24px",
            ),
            _progress_bar(),
            rx.match(
                FinetuneState.current_step,
                (1, _step1()),
                (2, _step2()),
                (3, _step3()),
                rx.text("Invalid step", color=c("text_muted")),
            ),
            spacing="0",
            width="100%",
            align_items="flex-start",
            padding="32px 24px",
        ),
        width="100%",
        max_width="1125px",
        margin="0 auto",
    )


# ── Page root ─────────────────────────────────────────────────────
def finetune_page() -> rx.Component:
    return rx.box(
        rx.cond(
            FinetuneState.current_step <= 3,
            _wizard_layout(),
            _workspace_layout(),
        ),
        width="100%",
        on_mount=[
            ExperimentState.load_runs,
            ModelRegistryState.load_models,
            FinetuneState.fetch_model_info,
        ],
    )
