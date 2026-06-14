"""TuneOS logo: static mark, wordmark, and lockups (docs/brand/BRAND_SPEC.md §2).

The mark is the resting ("mark") arrangement of the metaball animation, rendered
flat with the goo filter. It uses `currentColor`, so it follows the surrounding
theme — `--tune-ink` by default. Accent hues never touch the static mark (§4).
"""

from __future__ import annotations

import reflex as rx

from app.styles import c

# Chosen mark (option 5): two big blobs joined by a thin neck + three satellites.
# Must match MARK_BLOBS/MARK_NECKS in scripts/build_brand_assets.py.
_MARK_BLOBS = [(176, 184, 80), (304, 288, 80), (400, 104, 44), (104, 400, 48), (408, 408, 40)]
_MARK_NECKS = [(176, 184, 304, 288, 36)]
# Favicon reduction — the pair overlapping into a peanut + one satellite (§2b).
_FAVI_BLOBS = [(186, 206, 100), (300, 300, 100), (404, 110, 50)]

_GOO = (
    '<filter id="tune-goo" x="-30%" y="-30%" width="160%" height="160%">'
    '<feGaussianBlur in="SourceGraphic" stdDeviation="11" result="b"/>'
    '<feColorMatrix in="b" mode="matrix" '
    'values="1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 18 -7"/></filter>'
)


def _mark_svg(blobs, *, goo: bool, necks=()) -> str:
    lines = "".join(
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="currentColor" '
        f'stroke-width="{w}" stroke-linecap="round"/>'
        for (x1, y1, x2, y2, w) in necks
    )
    circles = "".join(f'<circle cx="{x}" cy="{y}" r="{r}"/>' for x, y, r in blobs)
    defs = f"<defs>{_GOO}</defs>" if goo else ""
    body = lines + circles
    grp = (
        f'<g filter="url(#tune-goo)" fill="currentColor">{body}</g>'
        if goo
        else f'<g fill="currentColor">{body}</g>'
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
    if use_favicon:
        svg = _mark_svg(_FAVI_BLOBS, goo=False)
    else:
        svg = _mark_svg(_MARK_BLOBS, goo=True, necks=_MARK_NECKS)
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
