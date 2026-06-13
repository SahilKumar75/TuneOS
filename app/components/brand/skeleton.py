"""Skeleton loaders — content placeholders shown while data/training loads.

A sweeping shimmer (keyframes `tune-shimmer` in app/styles.py) runs across a
two-stop gradient. Reduced-motion users get a static block. Shapes mirror the
real content they stand in for so the swap to live data doesn't reflow.

Primitives:
  skeleton_block(width, height, radius)
  skeleton_text(lines, last_width)
  skeleton_circle(size)
  skeleton_card()
  skeleton_table(rows, cols)
  skeleton_list(rows)
"""

from __future__ import annotations

import reflex as rx

from app.styles import c


def _shimmer_bg() -> rx.Var:
    """Theme-aware moving-highlight gradient used as the skeleton fill."""
    return rx.color_mode_cond(
        light="linear-gradient(90deg, #eceae4 25%, #f6f5f1 37%, #eceae4 63%)",
        dark="linear-gradient(90deg, #232323 25%, #2e2e2e 37%, #232323 63%)",
    )


def skeleton_block(
    width: str = "100%",
    height: str = "14px",
    radius: str = "6px",
) -> rx.Component:
    """One shimmering rectangle — the base primitive."""
    return rx.box(
        class_name="tune-skeleton",
        width=width,
        height=height,
        border_radius=radius,
        background=_shimmer_bg(),
    )


def skeleton_circle(size: str = "40px") -> rx.Component:
    return skeleton_block(width=size, height=size, radius="50%")


def skeleton_text(lines: int = 3, last_width: str = "60%") -> rx.Component:
    """Stack of text lines; the final line is shortened like real prose."""
    return rx.vstack(
        *[
            skeleton_block(
                width=last_width if i == lines - 1 else "100%",
                height="12px",
            )
            for i in range(lines)
        ],
        spacing="2",
        width="100%",
        align="start",
    )


def skeleton_card() -> rx.Component:
    """Avatar + heading + body — a generic content card placeholder."""
    return rx.box(
        rx.vstack(
            rx.hstack(
                skeleton_circle("44px"),
                rx.vstack(
                    skeleton_block(width="140px", height="13px"),
                    skeleton_block(width="90px", height="11px"),
                    spacing="2",
                    align="start",
                ),
                spacing="3",
                align="center",
                width="100%",
            ),
            rx.box(height="8px"),
            skeleton_text(3),
            spacing="3",
            width="100%",
            align="start",
        ),
        padding="16px",
        border="1px solid",
        border_color=c("border"),
        border_radius="12px",
        background=c("bg_card"),
        width="100%",
    )


def skeleton_table(rows: int = 5, cols: int = 4) -> rx.Component:
    """Header + body rows of shimmering cells."""
    def _row(is_header: bool) -> rx.Component:
        return rx.hstack(
            *[
                skeleton_block(
                    width="100%",
                    height="11px" if is_header else "13px",
                )
                for _ in range(cols)
            ],
            spacing="4",
            width="100%",
            padding_y="10px",
        )

    return rx.vstack(
        _row(is_header=True),
        rx.box(height="1px", width="100%", background=c("border")),
        *[_row(is_header=False) for _ in range(rows)],
        spacing="1",
        width="100%",
    )


def skeleton_list(rows: int = 4) -> rx.Component:
    """Vertical list of icon + two-line rows (sidebar / dataset list placeholder)."""
    return rx.vstack(
        *[
            rx.hstack(
                skeleton_circle("32px"),
                rx.vstack(
                    skeleton_block(width="70%", height="12px"),
                    skeleton_block(width="40%", height="10px"),
                    spacing="2",
                    align="start",
                    width="100%",
                ),
                spacing="3",
                align="center",
                width="100%",
                padding_y="8px",
            )
            for _ in range(rows)
        ],
        spacing="2",
        width="100%",
    )
