import reflex as rx


def training_page() -> rx.Component:
    return rx.box(on_mount=rx.redirect("/finetune"))
