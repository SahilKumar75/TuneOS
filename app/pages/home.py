import reflex as rx
from app.components.model_card import model_card

def home_page() -> rx.Component:
    return rx.container(
        rx.vstack(
            rx.heading("Open Source LLM Fine-Tuning", size="xl"),
            rx.text("Bring your dataset, pick a base model, and train an adapter in the cloud."),
            
            rx.heading("Select a Base Model", size="lg", margin_top="2em"),
            
            rx.grid(
                model_card(
                    "Mistral 7B", 
                    "mistralai/Mistral-7B-v0.1", 
                    "Primary target, well-tested with QLoRA"
                ),
                model_card(
                    "Llama 3 8B", 
                    "meta-llama/Meta-Llama-3-8B", 
                    "Requires HF token"
                ),
                model_card(
                    "Phi-3 Mini", 
                    "microsoft/Phi-3-mini-4k-instruct", 
                    "Fast, runs on smaller GPUs"
                ),
                model_card(
                    "Gemma 2B", 
                    "google/gemma-2b", 
                    "Good for low-VRAM environments"
                ),
                columns="2",
                spacing="4",
                width="100%"
            ),
            
            rx.button(
                "Next: Upload Dataset",
                on_click=rx.redirect("/upload"),
                size="lg",
                margin_top="2em",
                color_scheme="violet"
            ),
            align_items="center",
            padding="2em"
        )
    )
