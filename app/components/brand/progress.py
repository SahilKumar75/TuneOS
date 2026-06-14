"""Liquid progress bar — determinate + indeterminate (docs/brand/BRAND_SPEC.md §5).

Fill uses the blue→teal→purple liquid gradient: color = the system is doing real
work. For waiting with no measurable progress, use the indeterminate variant.
"""

from __future__ import annotations

import reflex as rx

from app.styles import c

_LIQUID = "linear-gradient(90deg, #5b8cff, #33d6c8, #a06bff)"


def liquid_progress(
    value: rx.Var | float,
    *,
    height: str = "8px",
    label: str | None = None,
    show_pct: bool = True,
) -> rx.Component:
    """Determinate bar. `value` is 0–100 (State var or literal)."""
    width = value.to_string() + "%" if isinstance(value, rx.Var) else f"{value}%"
    pct_text = value.to_string() + "%" if isinstance(value, rx.Var) else f"{value}%"
    bar = rx.box(
        rx.box(
            width=width,
            height="100%",
            background=_LIQUID,
            border_radius="999px",
            transition="width 0.4s cubic-bezier(0.65,0,0.35,1)",
        ),
        width="100%",
        height=height,
        background=c("bg_input"),
        border_radius="999px",
        overflow="hidden",
    )
    if not label and not show_pct:
        return bar
    head = rx.hstack(
        rx.text(label, font_size="0.8rem", color=c("text_secondary"), font_weight="500")
        if label
        else rx.fragment(),
        rx.spacer(),
        rx.cond(
            show_pct,
            rx.text(
                pct_text,
                font_size="0.8rem",
                color=c("text_primary"),
                font_weight="600",
                font_family="monospace",
            ),
            rx.fragment(),
        ),
        width="100%",
        align="center",
    )
    return rx.vstack(head, bar, spacing="2", width="100%")


def liquid_progress_indeterminate(height: str = "8px") -> rx.Component:
    """Indeterminate bar — a liquid sweep for unmeasurable waits."""
    return rx.box(
        rx.box(
            class_name="tune-indeterminate",
            height="100%",
            width="40%",
            background=_LIQUID,
            border_radius="999px",
        ),
        width="100%",
        height=height,
        background=c("bg_input"),
        border_radius="999px",
        overflow="hidden",
        position="relative",
    )
