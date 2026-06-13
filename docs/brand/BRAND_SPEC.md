# TuneOS — Brand + Logo Spec

> Build-ready identity reference. Source of truth for SVG assets and Reflex UI tokens.
> Status: codifies the existing metaball mark — do not redesign.

---

## 1. Brand essence

TuneOS is the operating system for LLM fine-tuning: technical, precise, and alive. The identity is a network of connection-dots that flow and merge like liquid metal — monochrome and exact at rest, electric and energetic in motion. It should feel less like a tool and more like a living system reorganizing itself.

---

## 2. Logo system

### (a) Mark — the metaball icon
The primary brand object. Black blobs on a notional **3×3 grid**, joined by smooth pinched necks via the goo filter (gaussian blur + alpha threshold), plus a few scattered satellite dots and an optional enclosed negative-space "hole."

- **Canonical mark:** 8–12 blobs in the `mark` arrangement (the resting state of the animation).
- **Render:** single fill color (`--tune-ink`), no stroke, no gradient (unless liquid-color mode is explicitly active).
- **Goo filter is mandatory** at sizes ≥ 32px. Below that, necks collapse visually — use favicon reduction instead.
- **Use when:** app icon, nav avatar, loader, splash, social avatar, OG image anchor.

### (b) Favicon — simplest legible reduction
At small sizes the goo filter and satellites become mud. Reduce to a **4–5 blob** cluster on a 2×2-ish footprint with one fat neck and one satellite dot.

- Drop the negative-space hole below 32px.
- Drop the goo filter below 24px; render blobs as overlapping circles with a hard union (no blur).
- Provide hand-tuned raster at 16/32/48; do not auto-downscale the full mark.
- **Use when:** browser tab, PWA icon, system tray, anywhere ≤ 48px.

### (c) Wordmark — "TuneOS"
Typographic, set to pair with the geometric/liquid mark.

- **Font family (recommended):** `Space Grotesk` (Google Fonts) — geometric, technical, slightly humanist; reads as "systems software."
- **Fallback / system pairing:** `"Space Grotesk", "Inter", ui-sans-serif, system-ui, sans-serif`. Use `Inter` alone if you want a more neutral, lower-personality wordmark.
- **Weight:** `600` (SemiBold) for the wordmark. Body/UI uses `Inter 400/500`.
- **Letter-spacing:** `-0.01em` (slight negative tracking — tightens the geometric forms).
- **Case:** `TuneOS` exactly — never `TuneOs`, `Tuneos`, `TUNEOS`.
- **"OS" differentiation (recommended):** keep one word, one weight, one color by default. For an emphasized lockup, set `OS` in `--tune-accent-blue` (light mode) / same accent (dark mode) while `Tune` stays `--tune-ink`. Do **not** differentiate `OS` by changing weight or size — color only, and only in the emphasized variant.

### (d) Lockup — mark + wordmark
Two locked configurations. Mark and wordmark optically aligned, never re-spaced ad hoc.

- **Horizontal:** mark left, wordmark right. Gap between mark and wordmark = `1× cap-height` of the wordmark. Vertically center the wordmark cap-height to the mark's optical center.
- **Stacked:** mark on top, wordmark centered below. Gap = `0.5× mark height`.
- **Sizing relationship:** mark height = `1.4× wordmark cap-height` (horizontal); mark height = `2× wordmark cap-height` (stacked).
- **Use horizontal** in top nav, headers, docs. **Use stacked** for splash, marketing hero, square contexts.

---

## 3. Clearspace & min sizes

Let `r` = radius of a primary blob in the mark.

- **Clearspace:** minimum margin around the mark/lockup on all sides = `2r` (≈ one blob diameter). No other element, text, or edge inside this zone.
- **Min mark size:** `24px` (favicon reduction applies below 32px; full mark not allowed below 24px).
- **Min wordmark size:** `16px` cap-height (below this, switch to mark-only).
- **Min lockup (horizontal):** `120px` wide.
- **Favicon raster sizes:** `16 / 32 / 48` px (hand-tuned), plus `180` (Apple touch) and `512` (PWA/maskable) from the full mark.

---

## 4. Color tokens

Developer-friendly names. Use as CSS custom properties / Reflex theme tokens. Mark is monochrome-first; accents are for liquid-color mode and UI energy only.

| Token | Light mode | Dark mode | Use |
|---|---|---|---|
| `--tune-surface` | `#f3f1ec` | `#141414` | Page background |
| `--tune-surface-raised` | `#faf9f5` | `#1c1c1c` | Cards, panels |
| `--tune-ink` | `#1c1c1c` | `#f3f1ec` | The mark, primary text |
| `--tune-ink-muted` | `#5c5a54` | `#a8a59d` | Secondary text, captions |
| `--tune-border` | `#e2dfd6` | `#2c2c2c` | Hairlines, dividers |
| `--tune-accent-blue` | `#5b8cff` | `#5b8cff` | Liquid accent / primary action |
| `--tune-accent-teal` | `#33d6c8` | `#33d6c8` | Liquid accent / success-energy |
| `--tune-accent-purple` | `#a06bff` | `#a06bff` | Liquid accent / processing-energy |
| `--tune-accent-blue-tint` | `#e9f0ff` | `#1a2540` | Accent fill backgrounds |
| `--tune-accent-teal-tint` | `#e0f9f6` | `#0f2e2b` | Accent fill backgrounds |
| `--tune-accent-purple-tint` | `#f1e9ff` | `#241a40` | Accent fill backgrounds |

**Liquid-color gradient** (the animated flow through blobs): linear/conic blend `blue → teal → purple → blue`, same three hues in both modes. Keep saturation as-is in dark mode (the off-black surface makes them glow correctly).

**Rule:** the mark is `--tune-ink` by default in both modes. Accent hues only appear when liquid-color mode is intentionally on.

---

## 5. Motion principles

The mark doubles as the app's loader/status object. It morphs between 8 arrangements: `mark, grid, scatter, wave, columns, rows, stream, merge`. The 12 blobs are **identity-stable** — they travel and resize, they never pop in/out.

1. **Easing feel — liquid, never linear.** Use `cubic-bezier(0.65, 0, 0.35, 1)` (ease-in-out, slightly weighted) for blob travel; necks should stretch and pinch, not snap. Transitions between arrangements run `600–900ms`. Idle ambient drift is slow: `4–8s` loops.

2. **Loading semantics map to arrangements:**
   - `merge` → **processing / committing** (writing a checkpoint, applying a step, saving). Blobs converge to a dense cluster.
   - `wave` → **streaming tokens** (generation/inference output). Continuous left-to-right ripple.
   - `stream` / `columns` / `rows` → **data flow** (loading a dataset, batching). Directional, conveyor-like.
   - `scatter` → **idle / empty state** (no job running, empty list). Loose, calm, low-energy drift.
   - `mark` → **resting / done** (success settle). Always return to `mark` when an operation completes.

3. **Merge vs scatter as energy direction.** Convergence (`merge`) reads as work/intensity; dispersion (`scatter`) reads as rest. Never sit in `merge` indefinitely — it implies an active process; if a job stalls, fall back to a slow `scatter` pulse.

4. **Speed range.** Ambient/idle: `4–8s` cycles. Active processing: `600–1200ms` cycles. Streaming `wave`: `1.5–2.5s` per ripple. Never faster than `500ms` per cycle — it stops reading as liquid and starts reading as flicker.

5. **Mono vs color.** Default loader is monochrome (`--tune-ink`). Switch to **liquid-color mode only for "alive / actively computing" states** — active training step, live token stream. Color = the system is doing real work. Mono = waiting, idle, or done.

6. **What NOT to do.** No bounce/overshoot/elastic easing (breaks the liquid metaphor — fluid doesn't spring). No rotating the whole mark as a spinner. No opacity-fade between arrangements (blobs travel, they don't dissolve). No more than one accent hue cross-fading at the *literal same instant* outside the defined `blue→teal→purple` flow order.

---

## 6. Do / Don't

**Do**
- Keep the mark single-color (`--tune-ink`) unless liquid-color mode is explicitly active.
- Respect `2r` clearspace and minimum sizes.
- Use the favicon reduction below 48px; hand-tune the raster.
- Return the loader to the `mark` arrangement on completion.
- Pair the wordmark in `Space Grotesk 600`, `-0.01em`.

**Don't**
- Don't stretch, skew, or non-uniformly scale the mark or lockup.
- Don't recolor the mark outside the defined tokens / liquid hues.
- Don't add drop shadows, outer glows, strokes, or bevels to the mark.
- Don't place the mark on a busy photo or low-contrast background — needs ≥ 4.5:1 against surface.
- Don't remove the goo filter at sizes where necks are visible (≥ 32px); don't keep it below 24px.
- Don't rewrite the casing: it's `TuneOS`, always.
- Don't differentiate `OS` by size or weight — accent color only, and only in the emphasized lockup.
- Don't use accent hues for the mark in static/idle contexts.
