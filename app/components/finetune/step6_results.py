"""Fine-tune wizard — Step 6: Results & evaluation."""

from __future__ import annotations

import reflex as rx

from app.components.finetune.shared import _card, _nav_buttons
from app.components.finetune.step5_train import _metric_tile
from app.state.experiment_state import ExperimentState, ModelRegistryState
from app.state.training_poller_state import TrainingPollerState
from app.styles import c


def step6_results() -> rx.Component:
    """Linear wizard layout for Step 6 (results & evaluation)."""
    return rx.vstack(
        rx.text(
            "Results & Evaluation",
            font_size="1.05rem",
            font_weight="600",
            color=c("text_primary"),
            margin_bottom="16px",
        ),
        # Eval metrics
        _card(
            rx.vstack(
                rx.hstack(
                    rx.text(
                        "Evaluation metrics",
                        font_size="0.9rem",
                        font_weight="600",
                        color=c("text_primary"),
                    ),
                    rx.spacer(),
                    rx.cond(
                        TrainingPollerState.eval_status == "idle",
                        rx.button(
                            "Run evaluation",
                            on_click=TrainingPollerState.run_eval,
                            size="2",
                            color_scheme="blue",
                            variant="soft",
                        ),
                        rx.badge(TrainingPollerState.eval_status, color_scheme="blue", size="1"),
                    ),
                    align="center",
                ),
                rx.cond(
                    TrainingPollerState.eval_status == "done",
                    rx.grid(
                        rx.vstack(
                            rx.text("Perplexity", font_size="0.72rem", color=c("text_muted")),
                            rx.text(
                                TrainingPollerState.eval_perplexity.to_string(),
                                font_size="1.8rem",
                                font_weight="700",
                                color=c("accent"),
                            ),
                            rx.text("Lower is better", font_size="0.7rem", color=c("text_muted")),
                            spacing="0",
                        ),
                        rx.vstack(
                            rx.text("ROUGE-1", font_size="0.72rem", color=c("text_muted")),
                            rx.text(
                                TrainingPollerState.eval_rouge1.to_string(),
                                font_size="1.8rem",
                                font_weight="700",
                                color=c("accent"),
                            ),
                            rx.text("Higher is better", font_size="0.7rem", color=c("text_muted")),
                            spacing="0",
                        ),
                        rx.vstack(
                            rx.text("BLEU", font_size="0.72rem", color=c("text_muted")),
                            rx.text(
                                TrainingPollerState.eval_bleu.to_string(),
                                font_size="1.8rem",
                                font_weight="700",
                                color=c("accent"),
                            ),
                            rx.text("Higher is better", font_size="0.7rem", color=c("text_muted")),
                            spacing="0",
                        ),
                        rx.vstack(
                            rx.text("What it means", font_size="0.72rem", color=c("text_muted")),
                            rx.text(
                                rx.cond(
                                    TrainingPollerState.eval_perplexity < 10,
                                    "Excellent — model learned the domain well",
                                    rx.cond(
                                        TrainingPollerState.eval_perplexity < 30,
                                        "Good — decent task alignment",
                                        "Try more epochs or a larger dataset",
                                    ),
                                ),
                                font_size="0.84rem",
                                color=c("text_secondary"),
                            ),
                            spacing="1",
                        ),
                        columns="4",
                        spacing="4",
                    ),
                    rx.fragment(),
                ),
                spacing="3",
            )
        ),
        # Inference tester
        _card(
            rx.vstack(
                rx.text(
                    "Test your model",
                    font_size="0.9rem",
                    font_weight="600",
                    color=c("text_primary"),
                    margin_bottom="8px",
                ),
                rx.cond(
                    TrainingPollerState.user_intent != "",
                    rx.text(
                        f"System context: {TrainingPollerState.user_intent}",
                        font_size="0.76rem",
                        color=c("text_muted"),
                        font_style="italic",
                    ),
                    rx.fragment(),
                ),
                rx.cond(
                    TrainingPollerState.test_chat_history.length() > 0,
                    rx.box(
                        rx.foreach(
                            TrainingPollerState.test_chat_history,
                            lambda msg: rx.box(
                                rx.text(
                                    msg.content,
                                    font_size="0.84rem",
                                    color=rx.cond(
                                        msg.role == "user",
                                        c("text_primary"),
                                        c("text_secondary"),
                                    ),
                                    padding="8px 12px",
                                    background=rx.cond(
                                        msg.role == "user",
                                        c("accent_soft"),
                                        c("bg_input"),
                                    ),
                                    border_radius="8px",
                                    align_self=rx.cond(
                                        msg.role == "user", "flex-end", "flex-start"
                                    ),
                                    max_width="80%",
                                ),
                                display="flex",
                                flex_direction=rx.cond(msg.role == "user", "row-reverse", "row"),
                                width="100%",
                                margin_bottom="6px",
                            ),
                        ),
                        width="100%",
                        max_height="300px",
                        overflow_y="auto",
                        padding="8px",
                        border="1px solid",
                        border_color=c("border"),
                        border_radius="8px",
                    ),
                    rx.fragment(),
                ),
                rx.hstack(
                    rx.input(
                        placeholder="Type a test message...",
                        value=TrainingPollerState.chat_input,
                        on_change=TrainingPollerState.set_chat_input,
                        on_key_down=TrainingPollerState.handle_chat_key,
                        flex="1",
                    ),
                    rx.button(
                        rx.cond(
                            TrainingPollerState.chat_loading,
                            rx.spinner(size="2"),
                            rx.icon("send", size=16),
                        ),
                        on_click=TrainingPollerState.send_test_chat,
                        disabled=TrainingPollerState.chat_loading,
                        color_scheme="blue",
                        size="2",
                    ),
                    spacing="2",
                ),
                rx.cond(
                    TrainingPollerState.chat_error != "",
                    rx.callout(TrainingPollerState.chat_error, color_scheme="red", size="1"),
                    rx.fragment(),
                ),
                spacing="3",
            )
        ),
        # Register to model registry
        rx.cond(
            (TrainingPollerState.training_status == "done")
            & (TrainingPollerState.experiment_id != ""),
            _card(
                rx.vstack(
                    rx.hstack(
                        rx.icon("bookmark", size=16, color=c("accent")),
                        rx.text(
                            "Register to model registry",
                            font_size="0.9rem",
                            font_weight="600",
                            color=c("text_primary"),
                        ),
                        spacing="2",
                        align="center",
                    ),
                    rx.text(
                        "Save this run under a name so you can promote it to production "
                        "or compare it against future runs.",
                        font_size="0.82rem",
                        color=c("text_secondary"),
                    ),
                    rx.cond(
                        TrainingPollerState.eval_status != "done",
                        rx.callout(
                            "Run evaluation first so accurate metrics are captured in the registry.",
                            color_scheme="amber",
                            size="1",
                        ),
                        rx.fragment(),
                    ),
                    rx.hstack(
                        rx.input(
                            placeholder="my-chatbot-v1",
                            value=ModelRegistryState.register_name,
                            on_change=ModelRegistryState.set_register_name,
                            flex="1",
                            disabled=TrainingPollerState.eval_status != "done",
                        ),
                        rx.button(
                            rx.cond(
                                ModelRegistryState.is_registering,
                                rx.hstack(rx.spinner(size="2"), rx.text("Saving…"), spacing="2"),
                                rx.text("Register"),
                            ),
                            on_click=ModelRegistryState.do_register(
                                TrainingPollerState.experiment_id,
                                TrainingPollerState.eval_perplexity,
                                TrainingPollerState.last_train_loss,
                            ),
                            disabled=ModelRegistryState.is_registering
                            | (TrainingPollerState.eval_status != "done"),
                            color_scheme="blue",
                            size="2",
                        ),
                        spacing="2",
                        width="100%",
                    ),
                    rx.cond(
                        ModelRegistryState.register_error != "",
                        rx.callout(ModelRegistryState.register_error, color_scheme="red", size="1"),
                        rx.fragment(),
                    ),
                    rx.cond(
                        ModelRegistryState.registered_run_id == TrainingPollerState.experiment_id,
                        rx.callout(
                            rx.hstack(
                                rx.icon("circle-check", size=14),
                                rx.text("Registered as " + ModelRegistryState.register_name),
                                spacing="2",
                            ),
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
        # Experiment comparison
        rx.cond(
            ExperimentState.completed_runs.length() > 1,
            _card(
                rx.vstack(
                    rx.text(
                        "Past runs with this setup",
                        font_size="0.9rem",
                        font_weight="600",
                        color=c("text_primary"),
                        margin_bottom="8px",
                    ),
                    rx.table.root(
                        rx.table.header(
                            rx.table.row(
                                rx.table.column_header_cell("Name"),
                                rx.table.column_header_cell("Model"),
                                rx.table.column_header_cell("Technique"),
                                rx.table.column_header_cell("LR"),
                                rx.table.column_header_cell("LoRA r"),
                                rx.table.column_header_cell("Batch"),
                                rx.table.column_header_cell("Epochs"),
                                rx.table.column_header_cell("Final Loss"),
                                rx.table.column_header_cell("Perplexity"),
                            )
                        ),
                        rx.table.body(
                            rx.foreach(
                                ExperimentState.completed_runs,
                                lambda r: rx.table.row(
                                    rx.table.cell(rx.text(r.name, font_size="0.8rem")),
                                    rx.table.cell(rx.text(r.model_id, font_size="0.8rem")),
                                    rx.table.cell(rx.text(r.technique, font_size="0.8rem")),
                                    rx.table.cell(rx.text(r.learning_rate, font_size="0.8rem")),
                                    rx.table.cell(
                                        rx.text(r.lora_r.to_string(), font_size="0.8rem")
                                    ),
                                    rx.table.cell(
                                        rx.text(r.batch_size.to_string(), font_size="0.8rem")
                                    ),
                                    rx.table.cell(
                                        rx.text(r.epochs.to_string(), font_size="0.8rem")
                                    ),
                                    rx.table.cell(
                                        rx.text(r.final_loss.to_string(), font_size="0.8rem")
                                    ),
                                    rx.table.cell(
                                        rx.text(r.perplexity.to_string(), font_size="0.8rem")
                                    ),
                                ),
                            )
                        ),
                        variant="surface",
                        size="1",
                        width="100%",
                    ),
                    spacing="2",
                )
            ),
            rx.fragment(),
        ),
        # Compare link — only meaningful when we have an experiment ID
        rx.cond(
            TrainingPollerState.experiment_id != "",
            rx.button(
                rx.hstack(
                    rx.icon("git-compare", size=14),
                    rx.text("Compare this run →"),
                    spacing="2",
                    align="center",
                ),
                on_click=rx.redirect(
                    "/compare?ids=" + TrainingPollerState.job_id
                ),
                variant="soft",
                color_scheme="blue",
                size="2",
            ),
            rx.fragment(),
        ),
        _nav_buttons(next_label="Next: Deploy →"),
        spacing="4",
        width="100%",
        align_items="flex-start",
    )


def step6_panel() -> rx.Component:
    """Workspace panel layout for Step 6 (results reveal below training area)."""
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.hstack(
                    rx.icon("bar-chart-2", size=14, color=c("accent")),
                    rx.text(
                        "Results & Evaluation",
                        font_size="0.82rem",
                        font_weight="600",
                        color=c("text_primary"),
                    ),
                    spacing="2",
                    align="center",
                ),
                rx.spacer(),
                rx.cond(
                    TrainingPollerState.experiment_id != "",
                    rx.button(
                        rx.hstack(
                            rx.icon("git-compare", size=13),
                            rx.text("Compare this run"),
                            spacing="1",
                            align="center",
                        ),
                        on_click=rx.redirect(
                            "/compare?ids=" + TrainingPollerState.job_id
                        ),
                        variant="ghost",
                        size="1",
                        color_scheme="blue",
                    ),
                    rx.fragment(),
                ),
                width="100%",
                align="center",
                margin_bottom="14px",
            ),
            rx.hstack(
                # Left col: eval metrics + inference tester
                rx.vstack(
                    _card(
                        rx.vstack(
                            rx.hstack(
                                rx.text(
                                    "Evaluation metrics",
                                    font_size="0.9rem",
                                    font_weight="600",
                                    color=c("text_primary"),
                                ),
                                rx.spacer(),
                                rx.cond(
                                    TrainingPollerState.eval_status == "idle",
                                    rx.button(
                                        "Run evaluation",
                                        on_click=TrainingPollerState.run_eval,
                                        size="2",
                                        color_scheme="blue",
                                        variant="soft",
                                    ),
                                    rx.badge(
                                        TrainingPollerState.eval_status,
                                        color_scheme="blue",
                                        size="1",
                                    ),
                                ),
                                align="center",
                            ),
                            rx.cond(
                                TrainingPollerState.eval_status == "done",
                                rx.grid(
                                    rx.vstack(
                                        rx.text(
                                            "Perplexity",
                                            font_size="0.72rem",
                                            color=c("text_muted"),
                                        ),
                                        rx.text(
                                            TrainingPollerState.eval_perplexity.to_string(),
                                            font_size="1.8rem",
                                            font_weight="700",
                                            color=c("accent"),
                                        ),
                                        rx.text(
                                            "Lower is better",
                                            font_size="0.7rem",
                                            color=c("text_muted"),
                                        ),
                                        spacing="0",
                                    ),
                                    rx.vstack(
                                        rx.text(
                                            "What it means",
                                            font_size="0.72rem",
                                            color=c("text_muted"),
                                        ),
                                        rx.text(
                                            rx.cond(
                                                TrainingPollerState.eval_perplexity < 10,
                                                "Excellent — model learned the domain well",
                                                rx.cond(
                                                    TrainingPollerState.eval_perplexity < 30,
                                                    "Good — decent task alignment",
                                                    "Try more epochs or a larger dataset",
                                                ),
                                            ),
                                            font_size="0.84rem",
                                            color=c("text_secondary"),
                                        ),
                                        spacing="1",
                                    ),
                                    columns="2",
                                    spacing="4",
                                ),
                                rx.fragment(),
                            ),
                            rx.cond(
                                TrainingPollerState.eval_status == "done",
                                rx.grid(
                                    _metric_tile(
                                        "ROUGE-1", TrainingPollerState.eval_rouge1.to_string()
                                    ),
                                    _metric_tile(
                                        "ROUGE-2", TrainingPollerState.eval_rouge2.to_string()
                                    ),
                                    _metric_tile(
                                        "ROUGE-L", TrainingPollerState.eval_rougeL.to_string()
                                    ),
                                    _metric_tile("BLEU", TrainingPollerState.eval_bleu.to_string()),
                                    _metric_tile(
                                        "METEOR", TrainingPollerState.eval_meteor.to_string()
                                    ),
                                    columns="3",
                                    spacing="3",
                                    width="100%",
                                ),
                                rx.fragment(),
                            ),
                            spacing="3",
                        )
                    ),
                    _card(
                        rx.vstack(
                            rx.text(
                                "Test your model",
                                font_size="0.9rem",
                                font_weight="600",
                                color=c("text_primary"),
                                margin_bottom="8px",
                            ),
                            rx.cond(
                                TrainingPollerState.user_intent != "",
                                rx.text(
                                    f"System context: {TrainingPollerState.user_intent}",
                                    font_size="0.76rem",
                                    color=c("text_muted"),
                                    font_style="italic",
                                ),
                                rx.fragment(),
                            ),
                            rx.cond(
                                TrainingPollerState.test_chat_history.length() > 0,
                                rx.box(
                                    rx.foreach(
                                        TrainingPollerState.test_chat_history,
                                        lambda msg: rx.box(
                                            rx.text(
                                                msg.content,
                                                font_size="0.84rem",
                                                color=rx.cond(
                                                    msg.role == "user",
                                                    c("text_primary"),
                                                    c("text_secondary"),
                                                ),
                                                padding="8px 12px",
                                                background=rx.cond(
                                                    msg.role == "user",
                                                    c("accent_soft"),
                                                    c("bg_input"),
                                                ),
                                                border_radius="8px",
                                                align_self=rx.cond(
                                                    msg.role == "user",
                                                    "flex-end",
                                                    "flex-start",
                                                ),
                                                max_width="80%",
                                            ),
                                            display="flex",
                                            flex_direction=rx.cond(
                                                msg.role == "user", "row-reverse", "row"
                                            ),
                                            width="100%",
                                            margin_bottom="6px",
                                        ),
                                    ),
                                    width="100%",
                                    max_height="200px",
                                    overflow_y="auto",
                                    padding="8px",
                                    border="1px solid",
                                    border_color=c("border"),
                                    border_radius="8px",
                                ),
                                rx.fragment(),
                            ),
                            rx.hstack(
                                rx.input(
                                    placeholder="Type a test message...",
                                    value=TrainingPollerState.chat_input,
                                    on_change=TrainingPollerState.set_chat_input,
                                    on_key_down=TrainingPollerState.handle_chat_key,
                                    flex="1",
                                ),
                                rx.button(
                                    rx.cond(
                                        TrainingPollerState.chat_loading,
                                        rx.spinner(size="2"),
                                        rx.icon("send", size=16),
                                    ),
                                    on_click=TrainingPollerState.send_test_chat,
                                    disabled=TrainingPollerState.chat_loading,
                                    color_scheme="blue",
                                    size="2",
                                ),
                                spacing="2",
                            ),
                            rx.cond(
                                TrainingPollerState.chat_error != "",
                                rx.callout(
                                    TrainingPollerState.chat_error, color_scheme="red", size="1"
                                ),
                                rx.fragment(),
                            ),
                            spacing="3",
                        )
                    ),
                    spacing="4",
                    flex="1",
                    min_width="0",
                ),
                # Right col: register + past runs
                rx.vstack(
                    rx.cond(
                        (TrainingPollerState.training_status == "done")
                        & (TrainingPollerState.experiment_id != ""),
                        _card(
                            rx.vstack(
                                rx.hstack(
                                    rx.icon("bookmark", size=16, color=c("accent")),
                                    rx.text(
                                        "Register to registry",
                                        font_size="0.9rem",
                                        font_weight="600",
                                        color=c("text_primary"),
                                    ),
                                    spacing="2",
                                    align="center",
                                ),
                                rx.text(
                                    "Save this run under a name for tracking and comparison.",
                                    font_size="0.82rem",
                                    color=c("text_secondary"),
                                ),
                                rx.cond(
                                    TrainingPollerState.eval_status != "done",
                                    rx.callout(
                                        "Run evaluation first for accurate metrics.",
                                        color_scheme="amber",
                                        size="1",
                                    ),
                                    rx.fragment(),
                                ),
                                rx.hstack(
                                    rx.input(
                                        placeholder="my-chatbot-v1",
                                        value=ModelRegistryState.register_name,
                                        on_change=ModelRegistryState.set_register_name,
                                        flex="1",
                                        disabled=TrainingPollerState.eval_status != "done",
                                    ),
                                    rx.button(
                                        rx.cond(
                                            ModelRegistryState.is_registering,
                                            rx.hstack(
                                                rx.spinner(size="2"),
                                                rx.text("Saving…"),
                                                spacing="2",
                                            ),
                                            rx.text("Register"),
                                        ),
                                        on_click=ModelRegistryState.do_register(
                                            TrainingPollerState.experiment_id,
                                            TrainingPollerState.eval_perplexity,
                                            TrainingPollerState.last_train_loss,
                                        ),
                                        disabled=ModelRegistryState.is_registering
                                        | (TrainingPollerState.eval_status != "done"),
                                        color_scheme="blue",
                                        size="2",
                                    ),
                                    spacing="2",
                                    width="100%",
                                ),
                                rx.cond(
                                    ModelRegistryState.register_error != "",
                                    rx.callout(
                                        ModelRegistryState.register_error,
                                        color_scheme="red",
                                        size="1",
                                    ),
                                    rx.fragment(),
                                ),
                                rx.cond(
                                    ModelRegistryState.registered_run_id
                                    == TrainingPollerState.experiment_id,
                                    rx.callout(
                                        rx.hstack(
                                            rx.icon("circle-check", size=14),
                                            rx.text(
                                                "Registered as " + ModelRegistryState.register_name
                                            ),
                                            spacing="2",
                                        ),
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
                    rx.cond(
                        ExperimentState.completed_runs.length() > 1,
                        _card(
                            rx.vstack(
                                rx.text(
                                    "Past runs",
                                    font_size="0.9rem",
                                    font_weight="600",
                                    color=c("text_primary"),
                                    margin_bottom="8px",
                                ),
                                rx.table.root(
                                    rx.table.header(
                                        rx.table.row(
                                            rx.table.column_header_cell("Name"),
                                            rx.table.column_header_cell("Technique"),
                                            rx.table.column_header_cell("LR"),
                                            rx.table.column_header_cell("Epochs"),
                                            rx.table.column_header_cell("Loss"),
                                            rx.table.column_header_cell("PPL"),
                                        )
                                    ),
                                    rx.table.body(
                                        rx.foreach(
                                            ExperimentState.completed_runs,
                                            lambda r: rx.table.row(
                                                rx.table.cell(rx.text(r.name, font_size="0.78rem")),
                                                rx.table.cell(
                                                    rx.text(r.technique, font_size="0.78rem")
                                                ),
                                                rx.table.cell(
                                                    rx.text(r.learning_rate, font_size="0.78rem")
                                                ),
                                                rx.table.cell(
                                                    rx.text(
                                                        r.epochs.to_string(),
                                                        font_size="0.78rem",
                                                    )
                                                ),
                                                rx.table.cell(
                                                    rx.text(
                                                        r.final_loss.to_string(),
                                                        font_size="0.78rem",
                                                    )
                                                ),
                                                rx.table.cell(
                                                    rx.text(
                                                        r.perplexity.to_string(),
                                                        font_size="0.78rem",
                                                    )
                                                ),
                                            ),
                                        )
                                    ),
                                    variant="surface",
                                    size="1",
                                    width="100%",
                                ),
                                spacing="2",
                            )
                        ),
                        rx.fragment(),
                    ),
                    spacing="4",
                    width="340px",
                    flex_shrink="0",
                    min_width="0",
                ),
                spacing="4",
                width="100%",
                align_items="flex-start",
            ),
            spacing="0",
            width="100%",
            align_items="flex-start",
        ),
        padding="20px",
        width="100%",
        border_top="2px solid",
        border_color=c("accent"),
        background=c("bg_card"),
        overflow_y="auto",
        max_height="50vh",
        class_name="step6-reveal",
    )
