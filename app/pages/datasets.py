"""TuneOS Datasets discovery page."""

import reflex as rx

from app.state.app_state import AppState
from app.styles import c

CATEGORIES = ["All", "NLP", "Code", "Math", "Science", "Chat", "Instruction"]

SAMPLE_DATASETS = [
    {
        "id": "alpaca",
        "name": "tatsu-lab/alpaca",
        "short": "Alpaca",
        "category": "Instruction",
        "rows": "52K rows",
        "desc": "Stanford Alpaca instruction-following dataset generated with GPT-3.",
        "license": "CC BY NC 4.0",
        "tags": ["instruction", "NLP"],
    },
    {
        "id": "dolly",
        "name": "databricks/databricks-dolly-15k",
        "short": "Dolly 15K",
        "category": "Instruction",
        "rows": "15K rows",
        "desc": "High-quality human-generated instruction-response pairs.",
        "license": "CC BY SA 3.0",
        "tags": ["instruction", "NLP"],
    },
    {
        "id": "openhermes",
        "name": "teknium/OpenHermes-2.5",
        "short": "OpenHermes 2.5",
        "category": "Chat",
        "rows": "1M rows",
        "desc": "Large synthetic chat dataset for instruction tuning.",
        "license": "MIT",
        "tags": ["chat", "NLP"],
    },
    {
        "id": "code_alpaca",
        "name": "sahil2801/CodeAlpaca-20k",
        "short": "CodeAlpaca",
        "category": "Code",
        "rows": "20K rows",
        "desc": "Code instruction-following dataset generated from GPT-3.",
        "license": "Apache 2.0",
        "tags": ["code"],
    },
    {
        "id": "math_instruct",
        "name": "TIGER-Lab/MathInstruct",
        "short": "MathInstruct",
        "category": "Math",
        "rows": "262K rows",
        "desc": "Math reasoning dataset with chain-of-thought solutions.",
        "license": "MIT",
        "tags": ["math"],
    },
    {
        "id": "sciq",
        "name": "allenai/sciq",
        "short": "SciQ",
        "category": "Science",
        "rows": "13.7K rows",
        "desc": "Science exam questions with supporting evidence.",
        "license": "CC BY NC 3.0",
        "tags": ["science", "QA"],
    },
    {
        "id": "sharegpt",
        "name": "anon8231489123/ShareGPT_Vicuna_unfiltered",
        "short": "ShareGPT",
        "category": "Chat",
        "rows": "90K rows",
        "desc": "Multi-turn ChatGPT conversations from ShareGPT.",
        "license": "Unknown",
        "tags": ["chat", "multi-turn"],
    },
    {
        "id": "python_code",
        "name": "iamtarun/python_code_instructions_18k_alpaca",
        "short": "Python Code 18K",
        "category": "Code",
        "rows": "18K rows",
        "desc": "Python code generation instructions in Alpaca format.",
        "license": "Apache 2.0",
        "tags": ["code", "python"],
    },
    {
        "id": "ultrachat",
        "name": "stingning/ultrachat",
        "short": "UltraChat",
        "category": "Chat",
        "rows": "1.5M rows",
        "desc": "Large-scale multi-turn chat dataset for fine-tuning.",
        "license": "CC BY NC 4.0",
        "tags": ["chat", "NLP"],
    },
]


def _dataset_card(ds: dict) -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.badge(ds["category"], variant="soft", color_scheme="blue"),
                rx.spacer(),
                rx.text(ds["rows"], font_size="0.78rem", color=c("text_muted")),
                align="center",
                width="100%",
            ),
            rx.text(
                ds["short"],
                font_size="0.95rem",
                font_weight="600",
                color=c("text_primary"),
            ),
            rx.text(
                ds["desc"],
                font_size="0.83rem",
                color=c("text_secondary"),
                line_height="1.4",
                overflow="hidden",
                display="-webkit-box",
                style={"-webkit-line-clamp": "2", "-webkit-box-orient": "vertical"},
            ),
            rx.hstack(
                rx.text(ds["license"], font_size="0.75rem", color=c("text_muted")),
                rx.spacer(),
                rx.button(
                    "Use",
                    on_click=AppState.set_hf_model(ds["name"]),
                    size="1",
                    variant="outline",
                    border_radius="999px",
                    cursor="pointer",
                    font_size="0.78rem",
                ),
                align="center",
                width="100%",
            ),
            spacing="2",
            align_items="flex-start",
            width="100%",
        ),
        padding="16px",
        background=c("bg_card"),
        border="1px solid",
        border_color=c("border"),
        border_radius="12px",
        cursor="pointer",
        _hover={"border_color": c("border_strong"), "background": c("hover")},
        transition="all 0.15s ease",
    )


def _category_item(label: str) -> rx.Component:
    return rx.text(
        label,
        font_size="0.88rem",
        color=c("text_secondary"),
        padding_x="12px",
        padding_y="7px",
        border_radius="8px",
        cursor="pointer",
        _hover={"background": c("hover"), "color": c("text_primary")},
    )


def datasets_page() -> rx.Component:
    return rx.hstack(
        rx.box(
            rx.vstack(
                rx.text(
                    "Categories",
                    font_size="0.78rem",
                    font_weight="500",
                    color=c("text_muted"),
                    padding_x="12px",
                    padding_top="8px",
                    padding_bottom="4px",
                ),
                *[_category_item(cat) for cat in CATEGORIES],
                spacing="1",
                align_items="flex-start",
                width="100%",
                padding_y="16px",
            ),
            width="180px",
            min_width="160px",
            border_right="1px solid",
            border_color=c("border"),
            height="100%",
            padding_x="8px",
        ),
        rx.box(
            rx.vstack(
                rx.hstack(
                    rx.heading(
                        "Datasets", font_size="1.4rem", font_weight="600", color=c("text_primary")
                    ),
                    rx.spacer(),
                    rx.button(
                        rx.icon("plus", size=15),
                        rx.text("Generate", font_size="0.88rem"),
                        variant="solid",
                        size="2",
                        background=rx.color_mode_cond(light="#171717", dark="#ededed"),
                        color=rx.color_mode_cond(light="#ffffff", dark="#171717"),
                        border_radius="8px",
                        cursor="pointer",
                    ),
                    align="center",
                    width="100%",
                ),
                rx.input(
                    placeholder="Search datasets...",
                    size="2",
                    width="100%",
                    max_width="420px",
                    background=c("bg_input"),
                    border="1px solid",
                    border_color=c("border"),
                    border_radius="8px",
                ),
                rx.grid(
                    *[_dataset_card(ds) for ds in SAMPLE_DATASETS],
                    columns="3",
                    spacing="4",
                    width="100%",
                ),
                spacing="5",
                align_items="flex-start",
                width="100%",
                padding="32px",
            ),
            flex="1",
            height="100%",
            overflow_y="auto",
            background=c("bg_primary"),
        ),
        spacing="0",
        width="100%",
        height="100vh",
        overflow="hidden",
        background=c("bg_primary"),
    )
