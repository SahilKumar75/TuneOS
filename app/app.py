"""
TuneOS — Main application entry point.
Configures the Reflex app with theming, global styles, and routes.
"""
import reflex as rx
from app.styles import STYLESHEETS, GLOBAL_STYLES
from app.components.layout import two_panel_layout
from app.pages.upload import upload_page
from app.pages.configure import configure_page
from app.pages.training import training_page
from app.pages.results import results_page


def index() -> rx.Component:
    """Landing page with the two-panel layout."""
    return two_panel_layout()


app = rx.App(
    theme=rx.theme(
        appearance="light",
        accent_color="blue",
        radius="medium",
        has_background=True,
    ),
    style=GLOBAL_STYLES,
    stylesheets=STYLESHEETS,
)

app.add_page(index, route="/", title="TuneOS — Fine-tune LLMs")
app.add_page(upload_page, route="/upload", title="Upload Dataset — TuneOS")
app.add_page(configure_page, route="/configure", title="Configure — TuneOS")
app.add_page(training_page, route="/training", title="Training — TuneOS")
app.add_page(results_page, route="/results", title="Results — TuneOS")

# Mount REST API endpoints
from app.api import router as api_router
app.api.include_router(api_router)
