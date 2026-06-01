"""Shared UI helpers used across all fine-tune wizard steps."""

from __future__ import annotations

import reflex as rx

from app.state.finetune_state import FinetuneState
from app.styles import c


def _card(*children, padding: str = "20px", width: str = "100%", **props) -> rx.Component:
    return rx.box(
        *children,
        background=c("bg_card"),
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


def _preview_table(rows: list, label: str = "Preview") -> rx.Component:
    return rx.vstack(
        rx.text(label, font_size="0.78rem", font_weight="500", color=c("text_muted")),
        rx.table.root(
            rx.table.header(
                rx.table.row(
                    rx.table.column_header_cell("Instruction"),
                    rx.table.column_header_cell("Output"),
                )
            ),
            rx.table.body(
                rx.foreach(
                    rows,
                    lambda row: rx.table.row(
                        rx.table.cell(
                            rx.text(
                                row["instruction"],
                                font_size="0.78rem",
                                overflow="hidden",
                                text_overflow="ellipsis",
                                white_space="nowrap",
                                max_width="300px",
                            )
                        ),
                        rx.table.cell(
                            rx.text(
                                row["output"],
                                font_size="0.78rem",
                                overflow="hidden",
                                text_overflow="ellipsis",
                                white_space="nowrap",
                                max_width="260px",
                            )
                        ),
                    ),
                )
            ),
            width="100%",
            variant="surface",
            size="1",
        ),
        width="100%",
        spacing="2",
    )
