# assets/

Static files served by Reflex at the site root (e.g. `assets/logo.svg` → `/logo.svg`).

## Reserved brand slots (not yet filled)

The UI already reserves space for the TuneOS logo so dropping the asset in causes
no layout shift:

| File | Used by | How to wire up |
|------|---------|----------------|
| `logo.svg` | sidebar brand mark | In `app/components/sidebar.py`, replace the inner box of `_brand_mark()` with `rx.image(src="/logo.svg", width="22px", height="22px")`. |
| `favicon.svg` | browser tab icon | In `app/app.py`, uncomment the `rx.el.link(rel="icon", ...)` line in `head_components`. |

Until the assets exist, the sidebar shows a neutral "T" monogram placeholder and no
custom favicon is set.
