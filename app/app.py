"""
TuneOS — Main application entry point.
Configures the Reflex app with theming, global styles, and routes.
"""

import logging

import reflex as rx

# #22 — structured JSON logging for the app process
try:
    from pythonjsonlogger import jsonlogger  # type: ignore[import]

    _handler = logging.StreamHandler()
    _handler.setFormatter(jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(name)s %(levelname)s %(message)s"
    ))
    logging.root.handlers = [_handler]
    logging.root.setLevel(logging.INFO)
except ImportError:
    logging.basicConfig(level=logging.INFO)

from app.components.layout import two_panel_layout
from app.pages.compare import compare_page
from app.pages.configure import configure_page
from app.pages.datasets import datasets_page
from app.pages.finetune import finetune_page
from app.pages.results import results_page
from app.pages.training import training_page
from app.pages.upload import upload_page
from app.state.finetune_state import FinetuneState
from app.state.theme_state import ThemeState
from app.styles import GLOBAL_STYLES, STYLESHEETS


def index() -> rx.Component:
    """Landing page with the two-panel layout."""
    return two_panel_layout()


_SYNC_THEME_SCRIPT = """
(function(){
  try {
    var saved = localStorage.getItem('theme');
    // First visit: seed key as 'system' so ThemeProvider follows OS preference.
    if (saved === null) {
      localStorage.setItem('theme', 'system');
      saved = 'system';
    }
    // Apply correct classes immediately to avoid flash before React hydrates.
    var dark = (saved === 'dark') ||
               (saved !== 'light' &&
                window.matchMedia('(prefers-color-scheme: dark)').matches);
    var mode = dark ? 'dark' : 'light';
    document.documentElement.classList.remove('light', 'dark');
    document.documentElement.classList.add(mode);
    document.documentElement.style.colorScheme = mode;
  } catch(e) {}
})();
"""

app = rx.App(
    style=GLOBAL_STYLES,
    stylesheets=STYLESHEETS,
    head_components=[
        rx.el.script(_SYNC_THEME_SCRIPT),
        # Logo/favicon slot — reserved. Drop `assets/favicon.svg` (see assets/README.md),
        # then uncomment to wire it up (no layout/SEO change until then):
        # rx.el.link(rel="icon", type="image/svg+xml", href="/favicon.svg"),
    ],
)

app.add_page(index, route="/", title="TuneOS — Fine-tune LLMs", on_load=ThemeState.init_theme)
app.add_page(upload_page, route="/upload", title="Upload Dataset — TuneOS")
app.add_page(configure_page, route="/configure", title="Configure — TuneOS")
app.add_page(training_page, route="/training", title="Training — TuneOS")
app.add_page(results_page, route="/results", title="Results — TuneOS")
app.add_page(datasets_page, route="/datasets", title="Datasets — TuneOS")
app.add_page(
    finetune_page,
    route="/finetune",
    title="Fine-tune — TuneOS",
    on_load=FinetuneState.resume_in_progress_job,  # #14 — restore poll loop after HF Spaces restart
)
app.add_page(compare_page, route="/compare", title="Compare — TuneOS")

# Mount REST API endpoints. Imported here, after page registration, to avoid
# a circular import between the Reflex app module and the API router.
from app.api import app_api  # noqa: E402

app._api.mount("/api", app_api)
