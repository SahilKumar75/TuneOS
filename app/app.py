import reflex as rx
from app.pages.home import home_page
from app.pages.upload import upload_page
from app.pages.configure import configure_page
from app.pages.training import training_page
from app.pages.results import results_page

app = rx.App()
app.add_page(home_page, route="/")
app.add_page(upload_page, route="/upload")
app.add_page(configure_page, route="/configure")
app.add_page(training_page, route="/training")
app.add_page(results_page, route="/results")
