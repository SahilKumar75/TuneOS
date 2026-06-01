"""TuneOS Datasets discovery page — search HF Hub, preview, use in fine-tune wizard."""

from __future__ import annotations

from typing import Any

import httpx
import reflex as rx

from app.state.finetune_state import FinetuneState
from app.styles import c

API_BASE = "http://localhost:8000"

CATEGORIES = ["All", "NLP", "Code", "Math", "Science", "Chat", "Instruction"]

# Curated starter cards (shown before search)
STARTER_DATASETS = [
    {"id": "tatsu-lab/alpaca", "short": "Alpaca", "category": "Instruction",
     "rows": "52K", "desc": "Stanford Alpaca instruction-following — the classic starting point.", "license": "CC BY NC 4.0"},
    {"id": "databricks/databricks-dolly-15k", "short": "Dolly 15K", "category": "Instruction",
     "rows": "15K", "desc": "High-quality human-generated instruction-response pairs.", "license": "CC BY SA 3.0"},
    {"id": "teknium/OpenHermes-2.5", "short": "OpenHermes 2.5", "category": "Chat",
     "rows": "1M", "desc": "Large synthetic chat dataset for instruction tuning.", "license": "MIT"},
    {"id": "sahil2801/CodeAlpaca-20k", "short": "CodeAlpaca", "category": "Code",
     "rows": "20K", "desc": "Code generation instructions in Alpaca format.", "license": "Apache 2.0"},
    {"id": "TIGER-Lab/MathInstruct", "short": "MathInstruct", "category": "Math",
     "rows": "262K", "desc": "Math reasoning with chain-of-thought solutions.", "license": "MIT"},
    {"id": "allenai/sciq", "short": "SciQ", "category": "Science",
     "rows": "13.7K", "desc": "Science exam questions with supporting evidence.", "license": "CC BY NC 3.0"},
    {"id": "iamtarun/python_code_instructions_18k_alpaca", "short": "Python Code 18K",
     "category": "Code", "rows": "18K", "desc": "Python code generation instructions.", "license": "Apache 2.0"},
    {"id": "stingning/ultrachat", "short": "UltraChat", "category": "Chat",
     "rows": "1.5M", "desc": "Large-scale multi-turn chat dataset.", "license": "CC BY NC 4.0"},
    {"id": "WizardLM/WizardLM_evol_instruct_70k", "short": "WizardLM 70K",
     "category": "Instruction", "rows": "70K", "desc": "Evolved instruction dataset for complex tasks.", "license": "Apache 2.0"},
]


class DatasetState(rx.State):
    search_query: str = ""
    search_results: list[dict[str, Any]] = []
    is_searching: bool = False
    selected_category: str = "All"

    # Preview panel
    preview_dataset_id: str = ""
    preview_columns: list[str] = []
    preview_rows: list[dict[str, Any]] = []
    is_loading_preview: bool = False
    preview_error: str = ""

    @rx.event
    def set_search_query(self, value: str):
        self.search_query = value

    @rx.event
    def set_category(self, cat: str):
        self.selected_category = cat

    @rx.event(background=True)
    async def search_datasets(self):
        async with self:
            self.is_searching = True
            self.search_results = []

        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.get(
                    f"{API_BASE}/api/datasets/search",
                    params={"q": self.search_query},
                )
            if resp.status_code == 200:
                async with self:
                    self.search_results = resp.json().get("results", [])
                    self.is_searching = False
            else:
                async with self:
                    self.is_searching = False
        except Exception:
            async with self:
                self.is_searching = False

    @rx.event(background=True)
    async def load_preview(self, dataset_id: str):
        async with self:
            self.preview_dataset_id = dataset_id
            self.is_loading_preview = True
            self.preview_error = ""
            self.preview_rows = []
            self.preview_columns = []

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(f"{API_BASE}/api/datasets/{dataset_id}/preview")
            if resp.status_code == 200:
                data = resp.json()
                async with self:
                    self.preview_columns = data.get("columns", [])
                    self.preview_rows = data.get("rows", [])
                    self.is_loading_preview = False
            else:
                async with self:
                    self.preview_error = "Failed to load preview"
                    self.is_loading_preview = False
        except Exception as exc:
            async with self:
                self.preview_error = str(exc)
                self.is_loading_preview = False

    @rx.event
    def use_in_finetune(self, dataset_id: str):
        return [
            FinetuneState.set_hub_dataset_id(dataset_id),
            rx.redirect("/finetune"),
        ]


# ── Components ────────────────────────────────────────────────────
def _card(*children, **props) -> rx.Component:
    return rx.box(
        *children,
        background=c("bg_card"),
        border="1px solid",
        border_color=c("border"),
        border_radius="12px",
        **props,
    )


def _dataset_card(ds: dict) -> rx.Component:
    is_previewing = DatasetState.preview_dataset_id == ds["id"]
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.badge(ds["category"], variant="soft", color_scheme="blue", size="1"),
                rx.spacer(),
                rx.text(ds["rows"] + " rows", font_size="0.74rem", color=c("text_muted")),
                align="center", width="100%",
            ),
            rx.text(ds["short"], font_size="0.92rem", font_weight="600", color=c("text_primary")),
            rx.text(ds["desc"], font_size="0.8rem", color=c("text_secondary"),
                    line_height="1.4", overflow="hidden",
                    style={"-webkit-line-clamp": "2", "-webkit-box-orient": "vertical",
                           "display": "-webkit-box"}),
            rx.hstack(
                rx.text(ds["license"], font_size="0.72rem", color=c("text_muted")),
                rx.spacer(),
                rx.hstack(
                    rx.button("Preview", size="1", variant="soft", color_scheme="gray",
                              on_click=DatasetState.load_preview(ds["id"])),
                    rx.button("Use in Fine-tune →", size="1", variant="solid",
                              color_scheme="blue",
                              on_click=DatasetState.use_in_finetune(ds["id"])),
                    spacing="1",
                ),
                align="center", width="100%",
            ),
            spacing="2", align_items="flex-start", width="100%",
        ),
        padding="16px",
        background=rx.cond(is_previewing, c("accent_soft"), c("bg_card")),
        border="1px solid",
        border_color=rx.cond(is_previewing, c("accent"), c("border")),
        border_radius="12px",
        _hover={"border_color": c("border_strong")},
        transition="all 0.15s ease",
    )


def _search_result_card(ds: dict) -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.text(ds["id"], font_size="0.88rem", font_weight="600",
                        color=c("text_primary")),
                rx.spacer(),
                rx.button("Use in Fine-tune →", size="1", variant="solid", color_scheme="blue",
                          on_click=DatasetState.use_in_finetune(ds["id"])),
                align="center", width="100%",
            ),
            rx.hstack(
                *[rx.badge(tag, size="1", variant="soft", color_scheme="gray")
                  for tag in (ds.get("tags") or [])[:4]],
                spacing="1", wrap="wrap",
            ),
            spacing="2", align_items="flex-start", width="100%",
        ),
        padding="14px",
        background=c("bg_card"),
        border="1px solid",
        border_color=c("border"),
        border_radius="10px",
    )


def _preview_panel() -> rx.Component:
    return rx.cond(
        DatasetState.preview_dataset_id != "",
        rx.box(
            rx.vstack(
                rx.hstack(
                    rx.icon("table", size=16, color=c("accent")),
                    rx.text(DatasetState.preview_dataset_id, font_weight="600",
                            font_size="0.86rem", color=c("text_primary")),
                    rx.spacer(),
                    rx.button("Use in Fine-tune →", size="2", color_scheme="blue",
                              on_click=DatasetState.use_in_finetune(
                                  DatasetState.preview_dataset_id)),
                    spacing="2", align="center", width="100%",
                ),
                rx.cond(
                    DatasetState.is_loading_preview,
                    rx.hstack(rx.spinner(size="2"),
                              rx.text("Loading preview...", font_size="0.82rem"),
                              spacing="2"),
                    rx.fragment(),
                ),
                rx.cond(
                    DatasetState.preview_error != "",
                    rx.callout(DatasetState.preview_error, color_scheme="red", size="1"),
                    rx.fragment(),
                ),
                rx.cond(
                    DatasetState.preview_rows.length() > 0,
                    rx.vstack(
                        rx.text("Columns: " + DatasetState.preview_columns.join(", "),
                                font_size="0.74rem", color=c("text_muted")),
                        rx.table.root(
                            rx.table.header(
                                rx.table.row(
                                    rx.foreach(
                                        DatasetState.preview_columns,
                                        lambda col: rx.table.column_header_cell(
                                            rx.text(col, font_size="0.76rem")
                                        ),
                                    )
                                )
                            ),
                            rx.table.body(
                                rx.foreach(
                                    DatasetState.preview_rows,
                                    lambda row: rx.table.row(
                                        rx.foreach(
                                            DatasetState.preview_columns,
                                            lambda col: rx.table.cell(
                                                rx.text(
                                                    row[col].to_string(),
                                                    font_size="0.76rem",
                                                    overflow="hidden",
                                                    text_overflow="ellipsis",
                                                    white_space="nowrap",
                                                    max_width="220px",
                                                )
                                            ),
                                        )
                                    ),
                                )
                            ),
                            variant="surface", size="1", width="100%",
                        ),
                        spacing="2",
                    ),
                    rx.fragment(),
                ),
                spacing="3",
            ),
            background=c("bg_card"),
            border="1px solid",
            border_color=c("border"),
            border_radius="12px",
            padding="16px",
            margin_top="16px",
        ),
        rx.fragment(),
    )


def _category_item(label: str) -> rx.Component:
    is_active = DatasetState.selected_category == label
    return rx.text(
        label,
        font_size="0.86rem",
        color=rx.cond(is_active, c("accent"), c("text_secondary")),
        font_weight=rx.cond(is_active, "600", "400"),
        padding_x="12px",
        padding_y="7px",
        border_radius="8px",
        cursor="pointer",
        background=rx.cond(is_active, c("accent_soft"), "transparent"),
        on_click=DatasetState.set_category(label),
        _hover={"background": c("hover"), "color": c("text_primary")},
    )


def datasets_page() -> rx.Component:
    return rx.hstack(
        # Category sidebar
        rx.box(
            rx.vstack(
                rx.text("Categories", font_size="0.74rem", font_weight="500",
                        color=c("text_muted"), padding_x="12px", padding_top="8px",
                        padding_bottom="4px"),
                *[_category_item(cat) for cat in CATEGORIES],
                spacing="1", align_items="flex-start", width="100%", padding_y="16px",
            ),
            width="180px", min_width="160px",
            border_right="1px solid", border_color=c("border"),
            height="100%", padding_x="8px",
        ),

        # Main area
        rx.box(
            rx.vstack(
                # Header
                rx.hstack(
                    rx.heading("Datasets", font_size="1.3rem", font_weight="600",
                               color=c("text_primary")),
                    rx.spacer(),
                    align="center", width="100%",
                ),

                # Search bar
                rx.hstack(
                    rx.input(
                        placeholder="Search HF Hub datasets — e.g. medical, code, chat...",
                        value=DatasetState.search_query,
                        on_change=DatasetState.set_search_query,
                        size="2", flex="1",
                    ),
                    rx.button(
                        rx.cond(DatasetState.is_searching,
                                rx.spinner(size="2"),
                                rx.hstack(rx.icon("search", size=14),
                                          rx.text("Search"), spacing="2")),
                        on_click=DatasetState.search_datasets,
                        disabled=DatasetState.is_searching,
                        color_scheme="blue", size="2",
                    ),
                    spacing="2", width="100%", max_width="560px",
                ),

                # Search results (when search has been triggered)
                rx.cond(
                    DatasetState.search_results.length() > 0,
                    rx.vstack(
                        rx.text(f"Search results ({DatasetState.search_results.length()})",
                                font_size="0.82rem", color=c("text_muted")),
                        rx.vstack(
                            rx.foreach(DatasetState.search_results, _search_result_card),
                            spacing="2", width="100%",
                        ),
                        spacing="2", width="100%",
                    ),
                    # Starter curated cards (default view)
                    rx.vstack(
                        rx.text("Curated datasets for fine-tuning",
                                font_size="0.82rem", color=c("text_muted")),
                        rx.grid(
                            *[_dataset_card(ds) for ds in STARTER_DATASETS],
                            columns="3", spacing="3", width="100%",
                        ),
                        spacing="2", width="100%",
                    ),
                ),

                # Preview panel (shown below list when a dataset is selected)
                _preview_panel(),

                spacing="5", align_items="flex-start", width="100%", padding="28px",
            ),
            flex="1", height="100%", overflow_y="auto", background=c("bg_primary"),
        ),

        spacing="0", width="100%", height="100vh", overflow="hidden",
        background=c("bg_primary"),
    )
