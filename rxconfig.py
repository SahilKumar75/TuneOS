import reflex as rx
from reflex_base.plugins.sitemap import SitemapPlugin

config = rx.Config(
    app_name="app",
    disable_plugins=[SitemapPlugin],
    plugins=[
        rx.plugins.RadixThemesPlugin(
            theme=rx.theme(
                appearance="inherit",
                accent_color="blue",
                radius="medium",
                has_background=True,
            )
        ),
    ],
)
