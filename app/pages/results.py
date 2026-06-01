import reflex as rx


def results_page() -> rx.Component:
    return rx.box(on_mount=rx.redirect("/finetune"))
