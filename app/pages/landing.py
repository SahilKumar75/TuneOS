"""TuneOS start screen with a focused software-style composer."""

import reflex as rx

from app.components.workspace_tabs import workspace_tab_bar
from app.state.app_state import AppState
from app.styles import c


def _dropdown_item(icon_name: str, label: str, on_click) -> rx.Component:
    return rx.hstack(
        rx.icon(icon_name, size=16, color=c("text_secondary")),
        rx.text(label, font_size="0.875rem", color=c("text_primary")),
        spacing="3",
        align="center",
        width="100%",
        padding_x="12px",
        padding_y="9px",
        border_radius="8px",
        cursor="pointer",
        _hover={"background": c("hover")},
        on_click=on_click,
    )


def _permission_selector() -> rx.Component:
    return rx.cond(
        AppState.show_permission_selector,
        rx.box(
            rx.vstack(
                _dropdown_item(
                    "chart-no-axes-column",
                    "Analytics",
                    AppState.select_permission_mode("analytics"),
                ),
                _dropdown_item("activity", "Training", AppState.select_permission_mode("training")),
                _dropdown_item(
                    "sliders-horizontal",
                    "Fine-tuning",
                    AppState.select_permission_mode("finetuning"),
                ),
                spacing="1",
                width="100%",
            ),
            position="absolute",
            top="100%",
            left="12px",
            margin_top="8px",
            z_index="20",
            width="210px",
            padding="6px",
            background=c("menu_bg"),
            border="1px solid",
            border_color=c("border_strong"),
            border_radius="12px",
            box_shadow=rx.color_mode_cond(
                light="0 14px 40px rgba(0,0,0,0.12)",
                dark="0 14px 40px rgba(0,0,0,0.38)",
            ),
        ),
    )


def _model_selector_item(icon_name: str, label: str, tab: str) -> rx.Component:
    active = AppState.active_tab == tab
    return rx.hstack(
        rx.icon(icon_name, size=15, color=c("text_secondary")),
        rx.text(label, font_size="0.85rem", color=c("text_primary"), flex="1"),
        rx.cond(active, rx.icon("check", size=15, color=c("accent")), rx.fragment()),
        spacing="2",
        align="center",
        width="100%",
        padding_x="12px",
        padding_y="9px",
        border_radius="8px",
        cursor="pointer",
        _hover={"background": c("hover")},
        on_click=AppState.select_tab_from_menu(tab),
    )


def _model_selector() -> rx.Component:
    return rx.cond(
        AppState.show_model_selector,
        rx.box(
            rx.vstack(
                _model_selector_item("globe", "Hugging Face", "huggingface"),
                _model_selector_item("git-branch", "GitHub", "github"),
                _model_selector_item("hard-drive", "Local", "local"),
                spacing="1",
                width="100%",
            ),
            position="absolute",
            top="100%",
            right="96px",
            margin_top="8px",
            z_index="20",
            width="190px",
            padding="6px",
            background=c("menu_bg"),
            border="1px solid",
            border_color=c("border_strong"),
            border_radius="12px",
            box_shadow=rx.color_mode_cond(
                light="0 14px 40px rgba(0,0,0,0.12)",
                dark="0 14px 40px rgba(0,0,0,0.38)",
            ),
        ),
    )


def _composer() -> rx.Component:
    return rx.box(
        _permission_selector(),
        _model_selector(),
        rx.vstack(
            rx.input(
                placeholder="Paste a Hugging Face link, GitHub URL, or local model path...",
                value=AppState.current_input_value,
                on_change=AppState.handle_input_change,
                variant="soft",
                size="3",
                width="100%",
                min_height="88px",
                background="transparent",
                border="none",
                color=c("text_primary"),
                font_size="0.98rem",
                _placeholder={"color": c("text_muted")},
                _focus={"outline": "none", "box_shadow": "none"},
            ),
            rx.hstack(
                rx.icon_button(
                    rx.icon("plus", size=17),
                    on_click=AppState.start_project,
                    variant="ghost",
                    size="2",
                    color=c("text_secondary"),
                    border_radius="8px",
                    cursor="pointer",
                    _hover={"background": c("hover"), "color": c("text_primary")},
                ),
                rx.spacer(),
                rx.icon_button(
                    rx.icon("arrow-up", size=20),
                    on_click=AppState.start_project,
                    variant="solid",
                    size="3",
                    background=rx.color_mode_cond(light="#171717", dark="#ededed"),
                    color=rx.color_mode_cond(light="#ffffff", dark="#171717"),
                    border_radius="999px",
                    cursor="pointer",
                    _hover={
                        "background": rx.color_mode_cond(light="#000000", dark="#ffffff"),
                    },
                ),
                spacing="2",
                align="center",
                width="100%",
            ),
            spacing="2",
            width="100%",
        ),
        position="relative",
        width="min(780px, calc(100vw - 80px))",
        background=c("input_bg"),
        border="1px solid",
        border_color=c("input_border"),
        border_radius="22px",
        padding="12px",
        box_shadow=rx.color_mode_cond(
            light="0 8px 28px rgba(0,0,0,0.06)",
            dark="0 8px 28px rgba(0,0,0,0.24)",
        ),
    )


def _preview_panel() -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.badge(AppState.preview_source_label, color_scheme="blue", variant="soft"),
                rx.spacer(),
                rx.button(
                    "Change",
                    on_click=AppState.cancel_preview,
                    variant="ghost",
                    size="2",
                    color=c("text_secondary"),
                    cursor="pointer",
                ),
                spacing="3",
                align="center",
                width="100%",
            ),
            rx.heading(
                AppState.preview_title,
                font_size="1.2rem",
                font_weight="600",
                color=c("text_primary"),
            ),
            rx.text(AppState.preview_meta, font_size="0.88rem", color=c("text_secondary")),
            rx.text(
                AppState.preview_summary,
                font_size="0.95rem",
                line_height="1.55",
                color=c("text_primary"),
            ),
            rx.text(AppState.preview_url, font_size="0.82rem", color=c("text_muted")),
            rx.hstack(
                rx.button(
                    "Yes, use this",
                    on_click=AppState.confirm_preview,
                    size="2",
                    background=rx.color_mode_cond(light="#171717", dark="#ededed"),
                    color=rx.color_mode_cond(light="#ffffff", dark="#171717"),
                    border_radius="999px",
                    cursor="pointer",
                ),
                rx.button(
                    "Cancel",
                    on_click=AppState.cancel_preview,
                    variant="outline",
                    size="2",
                    border_radius="999px",
                    cursor="pointer",
                ),
                spacing="3",
                align="center",
            ),
            spacing="3",
            width="100%",
        ),
        width="min(780px, calc(100vw - 80px))",
        padding="18px",
        background=c("bg_card"),
        border="1px solid",
        border_color=c("border"),
        border_radius="16px",
        box_shadow=rx.color_mode_cond(
            light="0 8px 28px rgba(0,0,0,0.06)",
            dark="0 8px 28px rgba(0,0,0,0.22)",
        ),
    )


def _preview_status() -> rx.Component:
    return rx.cond(
        AppState.preview_loading,
        rx.hstack(
            rx.spinner(size="2"),
            rx.text("Fetching link information...", font_size="0.92rem", color=c("text_secondary")),
            spacing="3",
            align="center",
            justify="center",
            width="min(780px, calc(100vw - 80px))",
            padding="14px",
        ),
        rx.cond(
            AppState.preview_ready,
            _preview_panel(),
            rx.cond(
                AppState.preview_error != "",
                rx.text(
                    AppState.preview_error,
                    font_size="0.92rem",
                    color=c("error"),
                    width="min(780px, calc(100vw - 80px))",
                    text_align="center",
                ),
                rx.fragment(),
            ),
        ),
    )


# ── Workspace Components ──────────────────────────────────────────


def _tag_pill(tag: rx.Var[str]) -> rx.Component:
    return rx.badge(
        tag,
        variant="soft",
        color_scheme="gray",
        size="1",
        font_size="0.7rem",
        border_radius="999px",
    )


def _info_row(icon_name: str, label: str, value: rx.Var[str]) -> rx.Component:
    return rx.cond(
        value != "",
        rx.hstack(
            rx.icon(icon_name, size=15, color=c("text_muted")),
            rx.text(label, font_size="0.85rem", color=c("text_muted"), min_width="100px"),
            rx.text(value, font_size="0.85rem", color=c("text_primary"), font_weight="500"),
            spacing="3",
            align="center",
            width="100%",
            padding_y="9px",
            border_bottom="1px solid",
            border_color=c("border"),
        ),
        rx.fragment(),
    )


def _action_tile(icon_name: str, label: str, on_click=None) -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.icon(icon_name, size=20, color=c("text_secondary")),
            rx.text(label, font_size="0.78rem", font_weight="500", color=c("text_primary")),
            spacing="1",
            align="center",
        ),
        padding="12px 8px",
        background=c("bg_card"),
        border="1px solid",
        border_color=c("border"),
        border_radius="10px",
        cursor="pointer",
        _hover={"border_color": c("accent"), "background": c("hover")},
        transition="all 0.15s ease",
        width="72px",
        text_align="center",
        on_click=on_click,
    )


def _action_icon_btn(icon_name: str) -> rx.Component:
    return rx.icon_button(
        rx.icon(icon_name, size=14),
        variant="ghost",
        size="1",
        color=c("text_muted"),
        border_radius="6px",
        cursor="pointer",
        _hover={"color": c("text_primary"), "background": c("hover")},
    )


def _chat_message(msg: rx.Var[dict[str, str]]) -> rx.Component:
    is_user = msg["role"] == "user"
    return rx.cond(
        is_user,
        # User message — right-aligned gray pill
        rx.hstack(
            rx.spacer(),
            rx.box(
                rx.text(
                    msg["text"], font_size="0.9rem", line_height="1.55", color=c("text_primary")
                ),
                padding="10px 16px",
                background=rx.color_mode_cond(light="#f0f0f0", dark="#2a2a2a"),
                border_radius="20px",
                max_width="72%",
            ),
            width="100%",
            align="start",
        ),
        # AI message — plain text left, action bar below
        rx.vstack(
            rx.text(msg["text"], font_size="0.9rem", line_height="1.65", color=c("text_primary")),
            rx.hstack(
                _action_icon_btn("copy"),
                spacing="0",
                align="center",
            ),
            spacing="1",
            align="start",
            width="100%",
            padding_left="2px",
        ),
    )


def _model_option(m: rx.Var[dict]) -> rx.Component:
    return rx.select.item(m["label"], value=m["id"])


def _chat_panel() -> rx.Component:
    return rx.box(
        rx.vstack(
            # Model selector header
            rx.hstack(
                rx.select.root(
                    rx.select.trigger(
                        placeholder="Auto (smart route)",
                        size="1",
                        variant="ghost",
                        color=c("text_secondary"),
                        font_size="0.78rem",
                        cursor="pointer",
                    ),
                    rx.select.content(
                        rx.foreach(AppState.CHAT_MODELS, _model_option),
                        position="popper",
                    ),
                    value=AppState.chat_model,
                    on_change=AppState.set_chat_model,
                ),
                rx.spacer(),
                rx.cond(
                    AppState.last_used_model != "",
                    rx.badge(
                        AppState.last_used_model,
                        variant="soft",
                        color_scheme="gray",
                        size="1",
                        font_size="0.68rem",
                        max_width="160px",
                        overflow="hidden",
                        text_overflow="ellipsis",
                        white_space="nowrap",
                    ),
                    rx.fragment(),
                ),
                align="center",
                width="100%",
                padding_bottom="10px",
                border_bottom="1px solid",
                border_color=c("border"),
            ),
            rx.cond(
                AppState.chat_messages.length() == 0,
                rx.vstack(
                    rx.vstack(
                        rx.icon("bot", size=28, color=c("text_muted")),
                        rx.text(
                            "Ask about this model",
                            font_size="0.92rem",
                            font_weight="500",
                            color=c("text_primary"),
                        ),
                        rx.text(
                            "Get help with fine-tuning, architecture, training configs, or dataset prep.",
                            font_size="0.82rem",
                            color=c("text_muted"),
                            text_align="center",
                            line_height="1.5",
                        ),
                        spacing="2",
                        align="center",
                    ),
                    rx.box(
                        rx.vstack(
                            *[
                                rx.text(
                                    hint,
                                    font_size="0.8rem",
                                    color=c("text_secondary"),
                                    line_height="1.4",
                                    width="100%",
                                    padding_y="8px",
                                    border_bottom=rx.cond(
                                        hint != "How much VRAM do I need to fine-tune this?",
                                        "1px solid",
                                        "none",
                                    ),
                                    border_color=c("border"),
                                    cursor="pointer",
                                    _hover={"color": c("text_primary")},
                                )
                                for hint in [
                                    "What LoRA rank works best for this model?",
                                    "Suggest a training config for my dataset size",
                                    "How much VRAM do I need to fine-tune this?",
                                ]
                            ],
                            spacing="0",
                            width="100%",
                        ),
                        width="100%",
                        padding="4px 14px",
                        background=c("bg_card"),
                        border="1px solid",
                        border_color=c("border"),
                        border_radius="12px",
                    ),
                    spacing="5",
                    align="center",
                    justify="center",
                    flex="1",
                    width="100%",
                    padding_y="24px",
                ),
                rx.vstack(
                    rx.foreach(AppState.chat_messages, _chat_message),
                    spacing="4",
                    width="100%",
                    flex="1",
                    overflow_y="auto",
                    padding_y="8px",
                    padding_x="4px",
                ),
            ),
            rx.box(
                rx.vstack(
                    rx.input(
                        placeholder="Ask anything",
                        value=AppState.chat_input,
                        on_change=AppState.set_chat_input,
                        on_key_down=AppState.handle_chat_key,
                        size="2",
                        width="100%",
                        background="transparent",
                        border="none",
                        box_shadow="none",
                        outline="none",
                        font_size="0.9rem",
                        color=c("text_primary"),
                        _placeholder={"color": c("text_muted")},
                        _focus={"outline": "none", "box_shadow": "none"},
                    ),
                    rx.hstack(
                        rx.icon_button(
                            rx.icon("plus", size=15),
                            variant="ghost",
                            size="1",
                            color=c("text_secondary"),
                            border_radius="6px",
                            cursor="pointer",
                            _hover={"background": c("hover"), "color": c("text_primary")},
                        ),
                        rx.spacer(),
                        rx.cond(
                            AppState.is_chat_loading,
                            rx.icon_button(
                                rx.icon("square", size=13, fill="currentColor"),
                                variant="solid",
                                size="2",
                                border_radius="999px",
                                background=rx.color_mode_cond(light="#171717", dark="#ededed"),
                                color=rx.color_mode_cond(light="#ffffff", dark="#171717"),
                                cursor="not-allowed",
                                disabled=True,
                            ),
                            rx.icon_button(
                                rx.icon("arrow-up", size=15),
                                on_click=AppState.send_chat_message,
                                variant="solid",
                                size="2",
                                border_radius="999px",
                                background=rx.color_mode_cond(light="#171717", dark="#ededed"),
                                color=rx.color_mode_cond(light="#ffffff", dark="#171717"),
                                cursor="pointer",
                                _hover={
                                    "background": rx.color_mode_cond(
                                        light="#000000", dark="#ffffff"
                                    ),
                                },
                            ),
                        ),
                        spacing="1",
                        align="center",
                        width="100%",
                    ),
                    spacing="1",
                    width="100%",
                ),
                width="100%",
                padding="10px 12px",
                background=c("bg_input"),
                border_radius="16px",
                border="none",
                box_shadow="none",
            ),
            spacing="3",
            height="100%",
            width="100%",
        ),
        width="380px",
        min_width="360px",
        height="100vh",
        padding_top="10px",
        padding_x="16px",
        padding_bottom="16px",
        background=c("bg_sidebar"),
        border_left="1px solid",
        border_color=c("border"),
    )


def _notebook_view() -> rx.Component:
    return rx.el.iframe(
        src="/notebook/index.html",
        style={
            "border": "none",
            "width": "100%",
            "height": "100%",
            "display": "block",
            "flex": "1",
            "min-height": "0",
        },
    )


def _workspace_content() -> rx.Component:
    return rx.hstack(
        rx.box(
            rx.vstack(
                # Tab bar (replaces old single-link top bar)
                workspace_tab_bar(),
                # Content area: dual-layer so the iframe is never unmounted
                rx.box(
                    # ── Model card layer ─────────────────────────────────
                    # Always in DOM; hidden via CSS when a notebook tab is active.
                    rx.box(
                        rx.vstack(
                            # Title row + action tiles
                            rx.hstack(
                                rx.vstack(
                                    rx.hstack(
                                        rx.badge(
                                            AppState.preview_source_label,
                                            color_scheme="blue",
                                            variant="soft",
                                            size="1",
                                        ),
                                        rx.cond(
                                            AppState.preview_license != "",
                                            rx.badge(
                                                AppState.preview_license,
                                                color_scheme="green",
                                                variant="soft",
                                                size="1",
                                            ),
                                            rx.fragment(),
                                        ),
                                        rx.cond(
                                            AppState.preview_params != "",
                                            rx.badge(
                                                AppState.preview_params + " params",
                                                color_scheme="orange",
                                                variant="soft",
                                                size="1",
                                            ),
                                            rx.fragment(),
                                        ),
                                        spacing="2",
                                        flex_wrap="wrap",
                                    ),
                                    rx.heading(
                                        AppState.preview_title,
                                        font_size="1.5rem",
                                        font_weight="700",
                                        line_height="1.2",
                                        color=c("text_primary"),
                                    ),
                                    rx.text(
                                        AppState.preview_pipeline,
                                        font_size="0.82rem",
                                        color=c("text_secondary"),
                                    ),
                                    spacing="2",
                                    flex="1",
                                ),
                                rx.hstack(
                                    _action_tile("cpu", "Train"),
                                    _action_tile("bar-chart-2", "Analyze"),
                                    _action_tile("refresh-cw", "Convert"),
                                    _action_tile(
                                        "notebook-pen",
                                        "Notebook",
                                        on_click=AppState.open_notebook_tab,
                                    ),
                                    spacing="2",
                                ),
                                align="start",
                                width="100%",
                                justify="between",
                            ),
                            # Tags + stats
                            rx.hstack(
                                rx.cond(
                                    AppState.preview_tags.length() > 0,
                                    rx.hstack(
                                        rx.foreach(AppState.preview_tags, _tag_pill),
                                        spacing="1",
                                        flex_wrap="wrap",
                                    ),
                                    rx.fragment(),
                                ),
                                rx.hstack(
                                    rx.cond(
                                        AppState.preview_downloads != "",
                                        rx.hstack(
                                            rx.icon("arrow-down-to-line", size=13, color="#4a9eff"),
                                            rx.text(
                                                AppState.preview_downloads,
                                                font_size="0.75rem",
                                                color=c("text_secondary"),
                                            ),
                                            spacing="1",
                                            align="center",
                                        ),
                                        rx.fragment(),
                                    ),
                                    rx.cond(
                                        AppState.preview_likes != "",
                                        rx.hstack(
                                            rx.icon("heart", size=13, color="#e85d75"),
                                            rx.text(
                                                AppState.preview_likes,
                                                font_size="0.75rem",
                                                color=c("text_secondary"),
                                            ),
                                            spacing="1",
                                            align="center",
                                        ),
                                        rx.fragment(),
                                    ),
                                    spacing="3",
                                    align="center",
                                    flex_shrink="0",
                                ),
                                align="center",
                                width="100%",
                                justify="between",
                                flex_wrap="wrap",
                                gap="2",
                            ),
                            rx.divider(border_color=c("border")),
                            # About
                            rx.vstack(
                                rx.text(
                                    "About",
                                    font_size="0.95rem",
                                    font_weight="600",
                                    color=c("text_primary"),
                                ),
                                rx.text(
                                    AppState.preview_summary,
                                    font_size="0.92rem",
                                    line_height="1.7",
                                    color=c("text_secondary"),
                                ),
                                spacing="3",
                                width="100%",
                            ),
                            rx.divider(border_color=c("border")),
                            # Left col (Model Details + Benchmark) + right col (Model Card)
                            rx.hstack(
                                rx.vstack(
                                    rx.vstack(
                                        rx.text(
                                            "Model Details",
                                            font_size="0.95rem",
                                            font_weight="600",
                                            color=c("text_primary"),
                                        ),
                                        rx.box(
                                            rx.vstack(
                                                _info_row(
                                                    "cpu",
                                                    "Architecture",
                                                    AppState.preview_architecture,
                                                ),
                                                _info_row(
                                                    "hash", "Parameters", AppState.preview_params
                                                ),
                                                _info_row(
                                                    "layers", "Pipeline", AppState.preview_pipeline
                                                ),
                                                _info_row(
                                                    "package", "Library", AppState.preview_library
                                                ),
                                                _info_row(
                                                    "hard-drive",
                                                    "Formats",
                                                    AppState.preview_formats,
                                                ),
                                                _info_row(
                                                    "file", "Files", AppState.preview_total_files
                                                ),
                                                _info_row(
                                                    "scale", "License", AppState.preview_license
                                                ),
                                                _info_row(
                                                    "calendar", "Created", AppState.preview_created
                                                ),
                                                _info_row(
                                                    "clock", "Updated", AppState.preview_updated
                                                ),
                                                spacing="0",
                                                width="100%",
                                            ),
                                            width="100%",
                                            padding="12px 16px",
                                            background=c("bg_card"),
                                            border="1px solid",
                                            border_color=c("border"),
                                            border_radius="10px",
                                        ),
                                        spacing="2",
                                        width="100%",
                                        align="start",
                                    ),
                                    rx.cond(
                                        AppState.preview_benchmark != "",
                                        rx.vstack(
                                            rx.text(
                                                "Benchmark Results",
                                                font_size="0.95rem",
                                                font_weight="600",
                                                color=c("text_primary"),
                                            ),
                                            rx.box(
                                                rx.markdown(AppState.preview_benchmark),
                                                width="100%",
                                                padding="12px 16px",
                                                background=c("bg_card"),
                                                border="1px solid",
                                                border_color=c("border"),
                                                border_radius="10px",
                                                overflow_x="hidden",
                                                class_name="bench-card",
                                            ),
                                            spacing="2",
                                            width="100%",
                                            align="start",
                                        ),
                                        rx.fragment(),
                                    ),
                                    spacing="4",
                                    width="420px",
                                    min_width="360px",
                                    flex_shrink="0",
                                    align="start",
                                ),
                                rx.cond(
                                    AppState.preview_readme != "",
                                    rx.vstack(
                                        rx.markdown(AppState.preview_readme_no_bench),
                                        spacing="2",
                                        flex="1",
                                        align="start",
                                        min_width="0",
                                        overflow_x="auto",
                                    ),
                                    rx.fragment(),
                                ),
                                spacing="5",
                                align="start",
                                width="100%",
                            ),
                            spacing="5",
                            width="100%",
                            padding="24px 28px",
                        ),
                        display=rx.cond(AppState.active_tab_is_notebook, "none", "block"),
                        width="100%",
                        height="100%",
                        overflow_y="auto",
                    ),
                    # ── Notebook iframe layer ─────────────────────────────
                    # Mounted once when the first notebook tab opens; never
                    # unmounted while a notebook tab exists → Pyodide stays alive.
                    rx.cond(
                        AppState.has_notebook_tab,
                        rx.box(
                            rx.el.iframe(
                                src="/notebook/index.html",
                                style={
                                    "border": "none",
                                    "width": "100%",
                                    "height": "100%",
                                },
                            ),
                            display=rx.cond(AppState.active_tab_is_notebook, "flex", "none"),
                            position="absolute",
                            top="0",
                            left="0",
                            width="100%",
                            height="100%",
                        ),
                        rx.fragment(),
                    ),
                    position="relative",
                    flex="1",
                    width="100%",
                    height="100%",
                    overflow="hidden",
                ),  # end content area
                spacing="0",
                width="100%",
                height="100%",
            ),
            flex="1",
            height="100vh",
            overflow="hidden",
            background=c("bg_primary"),
        ),
        _chat_panel(),
        spacing="0",
        width="100%",
        height="100vh",
        overflow="hidden",
    )


def _start_content() -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.heading(
                "What should we work on in TuneOS?",
                font_size="2.15rem",
                font_weight="500",
                line_height="1.2",
                color=c("text_primary"),
                text_align="center",
            ),
            _composer(),
            _preview_status(),
            spacing="6",
            align="center",
            justify="center",
            min_height="100vh",
            width="100%",
            padding_x="32px",
            padding_y="48px",
        ),
        background=c("bg_primary"),
        min_height="100vh",
        width="100%",
        overflow_y="auto",
    )


def landing_content() -> rx.Component:
    return rx.cond(
        AppState.workspace_active,
        _workspace_content(),
        _start_content(),
    )
