"""TuneOS logo: static mark, wordmark, and lockups (docs/brand/BRAND_SPEC.md §2).

The mark is the resting ("mark") arrangement of the metaball animation, rendered
flat with the goo filter. It uses `currentColor`, so it follows the surrounding
theme — `--tune-ink` by default. Accent hues never touch the static mark (§4).
"""

from __future__ import annotations

import reflex as rx

from app.styles import c

# Resting "mark" arrangement — must match BRAND in scripts/build_brand_assets.py.
_MARK_BLOBS = [
    (300, 90, 31), (360, 150, 31), (300, 210, 31), (240, 150, 31),
    (240, 210, 29), (185, 255, 29), (140, 315, 29), (240, 315, 29),
    (360, 255, 29), (95, 200, 12), (395, 95, 12), (395, 320, 12),
]
# Favicon reduction — 4 blobs + neck + satellite, no hole (§2b).
_FAVI_BLOBS = [
    (170, 170, 70), (310, 170, 70), (170, 310, 70), (310, 310, 70),
    (240, 240, 44), (400, 90, 30),
]

_GOO = (
    '<filter id="tune-goo" x="-30%" y="-30%" width="160%" height="160%">'
    '<feGaussianBlur in="SourceGraphic" stdDeviation="6" result="b"/>'
    '<feColorMatrix in="b" mode="matrix" '
    'values="1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 20 -9"/></filter>'
)


def _mark_svg(blobs, *, goo: bool) -> str:
    circles = "".join(f'<circle cx="{x}" cy="{y}" r="{r}"/>' for x, y, r in blobs)
    defs = f"<defs>{_GOO}</defs>" if goo else ""
    grp = (
        f'<g filter="url(#tune-goo)" fill="currentColor">{circles}</g>'
        if goo
        else f'<g fill="currentColor">{circles}</g>'
    )
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 480 480" '
        'width="100%" height="100%" style="display:block" role="img" '
        f'aria-label="TuneOS">{defs}{grp}</svg>'
    )


def tune_mark(size: str | int = 40, *, color: rx.Var | str | None = None) -> rx.Component:
    """Static metaball mark. `size` ≥ 32px keeps the necks; below that use favicon.

    `color` overrides the ink (defaults to theme `text_primary`).
    """
    dim = f"{size}px" if isinstance(size, int) else size
    use_favicon = isinstance(size, int) and size < 32
    svg = _mark_svg(_FAVI_BLOBS if use_favicon else _MARK_BLOBS, goo=not use_favicon)
    return rx.box(
        rx.html(svg),
        width=dim,
        height=dim,
        color=color if color is not None else c("text_primary"),
        flex_shrink="0",
        aria_label="TuneOS",
    )


_WORDMARK_FONT = '"Space Grotesk", "Inter", ui-sans-serif, system-ui, sans-serif'


def tune_wordmark(
    size: str = "1.5rem",
    *,
    accent_os: bool = False,
) -> rx.Component:
    """Typographic 'TuneOS'. Set `accent_os` to tint 'OS' in brand blue (§2c)."""
    return rx.box(
        rx.text(
            "Tune",
            rx.text(
                "OS",
                as_="span",
                color=c("accent_blue") if accent_os else "inherit",
            ),
            as_="span",
            font_family=_WORDMARK_FONT,
            font_weight="600",
            letter_spacing="-0.01em",
            font_size=size,
            color=c("text_primary"),
            line_height="1",
        ),
        display="inline-flex",
        align_items="center",
    )


def tune_lockup(
    *,
    stacked: bool = False,
    mark_size: int = 40,
    accent_os: bool = False,
) -> rx.Component:
    """Mark + wordmark. Horizontal for nav/headers, stacked for splash/hero (§2d)."""
    # Wordmark cap-height ≈ 0.7em; size relationship per spec.
    word_ratio = 0.5 if stacked else 1.0 / 1.4
    word_size = f"{mark_size * word_ratio / 16:.2f}rem"
    mark = tune_mark(mark_size)
    word = tune_wordmark(word_size, accent_os=accent_os)
    if stacked:
        return rx.vstack(mark, word, spacing="2", align="center")
    return rx.hstack(mark, word, spacing="3", align="center")
