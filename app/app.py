"""
TuneOS — Main application entry point.
Configures the Reflex app with theming, global styles, and routes.
"""

import reflex as rx

from app.components.layout import two_panel_layout
from app.pages.configure import configure_page
from app.pages.datasets import datasets_page
from app.pages.results import results_page
from app.pages.training import training_page
from app.pages.upload import upload_page
from app.styles import GLOBAL_STYLES, STYLESHEETS


def index() -> rx.Component:
    """Landing page with the two-panel layout."""
    return two_panel_layout()


_SYNC_THEME_SCRIPT = """
(function(){
  try {
    var dark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    var mode = dark ? 'dark' : 'light';
    // Overwrite whatever Reflex / Radix stored previously
    localStorage.setItem('color_mode', mode);
    document.cookie = 'color_mode=' + mode + ';path=/;SameSite=Lax';
    // Immediately apply so there is zero flash
    document.documentElement.classList.toggle('dark', dark);
    document.documentElement.style.colorScheme = mode;
    // Keep in sync if the user changes their OS theme while the tab is open
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function(e){
      var m = e.matches ? 'dark' : 'light';
      localStorage.setItem('color_mode', m);
      document.cookie = 'color_mode=' + m + ';path=/;SameSite=Lax';
      document.documentElement.classList.toggle('dark', e.matches);
      document.documentElement.style.colorScheme = m;
    });
  } catch(e) {}
})();
"""

app = rx.App(
    theme=rx.theme(
        appearance="inherit",
        accent_color="blue",
        radius="medium",
        has_background=True,
    ),
    style=GLOBAL_STYLES,
    stylesheets=STYLESHEETS,
    head_components=[rx.el.script(_SYNC_THEME_SCRIPT)],
)

app.add_page(index, route="/", title="TuneOS — Fine-tune LLMs")
app.add_page(upload_page, route="/upload", title="Upload Dataset — TuneOS")
app.add_page(configure_page, route="/configure", title="Configure — TuneOS")
app.add_page(training_page, route="/training", title="Training — TuneOS")
app.add_page(results_page, route="/results", title="Results — TuneOS")
app.add_page(datasets_page, route="/datasets", title="Datasets — TuneOS")

# Mount REST API endpoints. Imported here, after page registration, to avoid
# a circular import between the Reflex app module and the API router.
from app.api import app_api  # noqa: E402

app._api.mount("/api", app_api)
