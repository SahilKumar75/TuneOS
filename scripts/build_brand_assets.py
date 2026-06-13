"""Generate TuneOS brand SVG assets (animated metaball loader, mark, favicon).

The loader is a self-contained SMIL animation: 12 identity-stable "blobs" travel
between 8 arrangements, merging and splitting through a goo filter (blur + alpha
threshold). No JavaScript — it loops natively in any browser, <img>, or favicon,
which is what makes it safe to embed in Reflex (rx.html does not execute scripts).

Run:  python scripts/build_brand_assets.py
Emits to assets/:
  tuneos-loader.svg        mono, animated
  tuneos-loader-color.svg  liquid-gradient, animated
  tuneos-mark.svg          static brand mark (currentColor)
  tuneos-favicon.svg       5-blob reduction (static)
"""

from __future__ import annotations

import math
from pathlib import Path

ASSETS = Path(__file__).resolve().parents[1] / "assets"
N = 12
C = 240          # canvas centre (480x480 box)
EASE = "0.65 0 0.35 1"   # cubic ease-in-out — slows to a near-pause at each state


# ── State generators: each returns 12 (x, y, r) tuples ────────────────────────
def ring(rad=120, r=26):
    return [
        (
            C + rad * math.cos(math.radians(-90 + i * 30)),
            C + rad * math.sin(math.radians(-90 + i * 30)),
            r,
        )
        for i in range(N)
    ]


def grid():
    gx = gy = [120, 240, 360]
    a = [(gx[i % 3], gy[i // 3], 30) for i in range(9)]
    a += [(60, 60, 11), (420, 60, 11), (60, 420, 11)]
    return a


def columns():
    colx, rowy = [150, 240, 330], [110, 190, 290, 370]
    return [(colx[i % 3], rowy[i // 3], 27) for i in range(N)]


def rows():
    rowy, colx = [120, 240, 360], [110, 200, 280, 370]
    return [(colx[i // 3], rowy[i % 3], 27) for i in range(N)]


def wave():
    return [(80 + i * 30, C + 46 * math.sin(i * 0.62), 17) for i in range(N)]


def diagonal():
    out = []
    for i in range(N):
        t = i / (N - 1)
        out.append((120 + t * 240, 90 + t * 300, 32 - 13 * abs(t - 0.5) * 2))
    return out


def cluster():
    return [
        (C + 10 * math.sqrt(i) * math.cos(i * 2.399),
         C + 10 * math.sqrt(i) * math.sin(i * 2.399), 25)
        for i in range(N)
    ]


SCATTER = [
    (120, 90, 22), (220, 70, 13), (350, 100, 25), (410, 200, 12),
    (340, 300, 22), (240, 250, 16), (150, 320, 25), (80, 230, 13),
    (200, 180, 11), (300, 175, 19), (100, 150, 10), (380, 360, 12),
]

BRAND = [
    (300, 90, 31), (360, 150, 31), (300, 210, 31), (240, 150, 31),
    (240, 210, 29), (185, 255, 29), (140, 315, 29), (240, 315, 29),
    (360, 255, 29), (95, 200, 12), (395, 95, 12), (395, 320, 12),
]

STATES = [BRAND, grid(), SCATTER, wave(), columns(), rows(), diagonal(), cluster()]


def _seq(i, idx):
    """SMIL values list for blob i, attribute idx (0=x,1=y,2=r), looping home."""
    return ";".join(
        [f"{s[i][idx]:.1f}" for s in STATES] + [f"{STATES[0][i][idx]:.1f}"]
    )


def _loader(color: bool) -> str:
    splines = ";".join([EASE] * len(STATES))
    keytimes = ";".join(f"{k/len(STATES):.4f}" for k in range(len(STATES) + 1))
    fill = "url(#lg)" if color else "currentColor"
    grad = ""
    if color:
        grad = (
            '<linearGradient id="lg" x1="0" y1="0" x2="1" y2="1">'
            '<animateTransform attributeName="gradientTransform" type="rotate"'
            ' from="0 .5 .5" to="360 .5 .5" dur="14s" repeatCount="indefinite"/>'
            '<stop offset="0" stop-color="#5b8cff">'
            '<animate attributeName="stop-color" values="#5b8cff;#33d6c8;#a06bff;#5b8cff" dur="7s" repeatCount="indefinite"/></stop>'
            '<stop offset="0.55" stop-color="#33d6c8">'
            '<animate attributeName="stop-color" values="#33d6c8;#a06bff;#5b8cff;#33d6c8" dur="7s" repeatCount="indefinite"/></stop>'
            '<stop offset="1" stop-color="#a06bff">'
            '<animate attributeName="stop-color" values="#a06bff;#5b8cff;#33d6c8;#a06bff" dur="7s" repeatCount="indefinite"/></stop>'
            "</linearGradient>"
        )
    circles = []
    for i in range(N):
        cx = _seq(i, 0)
        cy = _seq(i, 1)
        cr = _seq(i, 2)
        base = STATES[0][i]
        # tiny perpetual drift so the mark breathes even while a state "holds"
        dx = 4 * math.cos(i * 1.7)
        dy = 4 * math.sin(i * 2.1)
        wob = (
            f'<animateTransform attributeName="transform" type="translate" additive="sum" '
            f'values="0 0;{dx:.1f} {dy:.1f};0 0;{-dx:.1f} {-dy:.1f};0 0" '
            f'dur="{5 + i % 4}s" repeatCount="indefinite"/>'
        )
        anim = (
            f'<animate attributeName="cx" values="{cx}" keyTimes="{keytimes}" '
            f'calcMode="spline" keySplines="{splines}" dur="20s" repeatCount="indefinite"/>'
            f'<animate attributeName="cy" values="{cy}" keyTimes="{keytimes}" '
            f'calcMode="spline" keySplines="{splines}" dur="20s" repeatCount="indefinite"/>'
            f'<animate attributeName="r" values="{cr}" keyTimes="{keytimes}" '
            f'calcMode="spline" keySplines="{splines}" dur="20s" repeatCount="indefinite"/>'
        )
        circles.append(
            f'<circle cx="{base[0]:.1f}" cy="{base[1]:.1f}" r="{base[2]:.1f}">{wob}{anim}</circle>'
        )
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 480 480" '
        'width="480" height="480" role="img" aria-label="TuneOS loading">'
        "<defs>"
        '<filter id="goo" x="-30%" y="-30%" width="160%" height="160%">'
        '<feGaussianBlur in="SourceGraphic" stdDeviation="6" result="b"/>'
        '<feColorMatrix in="b" mode="matrix" '
        'values="1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 20 -9"/>'
        "</filter>"
        f"{grad}"
        "</defs>"
        f'<g filter="url(#goo)" fill="{fill}">{"".join(circles)}</g>'
        "</svg>"
    )


def _static(positions, *, label, vbox=480, blob="currentColor", goo=True) -> str:
    circles = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}"/>' for (x, y, r) in positions
    )
    if goo:
        defs = (
            "<defs>"
            '<filter id="goo" x="-30%" y="-30%" width="160%" height="160%">'
            '<feGaussianBlur in="SourceGraphic" stdDeviation="6" result="b"/>'
            '<feColorMatrix in="b" mode="matrix" '
            'values="1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 20 -9"/>'
            "</filter></defs>"
        )
        grp = f'<g filter="url(#goo)" fill="{blob}" color="#1c1c1c">{circles}</g>'
    else:
        # Hard union of overlapping circles — survives tiny favicon sizes (§2b).
        defs = ""
        grp = f'<g fill="{blob}" color="#1c1c1c">{circles}</g>'
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {vbox} {vbox}" '
        f'role="img" aria-label="{label}">{defs}{grp}</svg>'
    )


# Favicon: simplest legible reduction — 2x2 block + neck + one satellite.
FAVICON = [
    (170, 170, 70), (310, 170, 70), (170, 310, 70), (310, 310, 70),
    (240, 240, 44), (400, 90, 30),
]


def main():
    (ASSETS / "tuneos-loader.svg").write_text(_loader(color=False))
    (ASSETS / "tuneos-loader-color.svg").write_text(_loader(color=True))
    (ASSETS / "tuneos-mark.svg").write_text(_static(BRAND, label="TuneOS"))
    (ASSETS / "tuneos-favicon.svg").write_text(
        _static(FAVICON, label="TuneOS", goo=False)
    )
    print("wrote 4 assets to", ASSETS)


if __name__ == "__main__":
    main()
