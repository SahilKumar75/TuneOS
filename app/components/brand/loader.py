"""TuneOS metaball loaders.

The animation is a self-contained SMIL SVG (assets/tuneos-loader*.svg, generated
by scripts/build_brand_assets.py). SMIL loops with no JavaScript, so it survives
rx.html's innerHTML injection — which silently drops <script> tags.

Semantics (docs/brand/BRAND_SPEC.md §5):
  - mono  → waiting / idle / done
  - color → the system is actively computing (live training step, token stream)

Components:
  metaball_loader(size, color, label)  inline, centered
  metaball_overlay(label, color)       full-viewport blocking overlay
  metaball_spinner(size, color)        small inline busy indicator
"""

from __future__ import annotations

from pathlib import Path

import reflex as rx

from app.styles import c

_ASSETS = Path(__file__).resolve().parents[3] / "assets"


def _svg(name: str) -> str:
    raw = (_ASSETS / name).read_text()
    # Let the wrapping box drive the size; keep the SVG square + responsive.
    return raw.replace(
        'width="480" height="480"',
        'width="100%" height="100%" style="display:block"',
        1,
    )


# Read once at import; the markup is static (animation lives in SMIL attrs).
_MONO = _svg("tuneos-loader.svg")
_COLOR = _svg("tuneos-loader-color.svg")


def metaball_loader(
    size: str | int = 120,
    *,
    color: bool = False,
    label: str | None = None,
) -> rx.Component:
    """Centered liquid metaball loader.

    Args:
        size: px (int) or any CSS length for the square mark.
        color: liquid-gradient mode (use only while actively computing).
        label: optional status caption shown below the mark.
    """
    dim = f"{size}px" if isinstance(size, int) else size
    mark = rx.box(
        rx.html(_COLOR if color else _MONO),
        width=dim,
        height=dim,
        # Mono inherits this color via `currentColor`; color mode ignores it.
        color=c("text_primary"),
        flex_shrink="0",
    )
    if not label:
        return mark
    return rx.vstack(
        mark,
        rx.text(label, font_size="0.85rem", color=c("text_secondary"), font_weight="500"),
        spacing="3",
        align="center",
        justify="center",
    )


def metaball_spinner(size: str | int = 40, *, color: bool = True) -> rx.Component:
    """Small inline busy indicator. Defaults to color (it implies live work)."""
    return metaball_loader(size, color=color)


def metaball_overlay(
    label: str | None = "Working…",
    *,
    color: bool = True,
    visible: rx.Var | bool = True,
) -> rx.Component:
    """Full-viewport blocking overlay with a centered loader.

    Pass `visible` a boolean State var to toggle it (rendered only when truthy).
    """
    panel = rx.box(
        rx.center(
            metaball_loader(140, color=color, label=label),
            width="100%",
            height="100%",
        ),
        position="fixed",
        inset="0",
        z_index="3000",
        background=rx.color_mode_cond(
            light="rgba(255,255,255,0.72)", dark="rgba(10,10,10,0.72)"
        ),
        backdrop_filter="blur(6px)",
    )
    if visible is True:
        return panel
    return rx.cond(visible, panel, rx.fragment())
