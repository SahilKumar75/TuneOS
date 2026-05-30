import os

import reflex as rx

from app.state.model_state import ModelState


class UploadState(rx.State):
    """The app state for the upload page."""

    # The images to show.
    file_path: str = ""
    is_uploading: bool = False

    async def handle_upload(self, files: list[rx.UploadFile]):
        """Handle the upload of file(s)."""
        self.is_uploading = True

        # We only take the first file for simplicity
        if not files:
            self.is_uploading = False
            return

        file = files[0]
        upload_data = await file.read()

        # Save to storage directory
        dataset_dir = os.getenv("DATASET_DIR", "./storage/datasets")
        os.makedirs(dataset_dir, exist_ok=True)

        outfile_path = os.path.join(dataset_dir, file.filename)

        with open(outfile_path, "wb") as f:
            f.write(upload_data)

        self.file_path = outfile_path
        self.is_uploading = False

        # Update model state with the dataset path
        return ModelState.set_dataset_path(outfile_path)


def upload_page() -> rx.Component:
    return rx.container(
        rx.vstack(
            rx.heading("Upload Dataset", size="8"),
            rx.text("Upload a JSONL or CSV file with 'instruction' and 'output' columns."),
            rx.upload(
                rx.vstack(
                    rx.button(
                        "Select File",
                        color_scheme="blue",
                        variant="outline",
                    ),
                    rx.text("Drag and drop files here or click to select files"),
                ),
                id="upload_dataset",
                multiple=False,
                accept={"application/json": [".jsonl", ".json"], "text/csv": [".csv"]},
                max_files=1,
                padding="2em",
                border="1px dashed var(--gray-4)",
                border_radius="md",
                width="100%",
            ),
            rx.button(
                "Upload",
                on_click=UploadState.handle_upload(rx.upload_files(upload_id="upload_dataset")),
                color_scheme="green",
                margin_top="1em",
            ),
            rx.cond(
                UploadState.file_path != "",
                rx.text(
                    f"Successfully uploaded: {UploadState.file_path}",
                    color="green",
                    font_weight="bold",
                ),
            ),
            rx.cond(
                UploadState.is_uploading,
                rx.spinner(),
            ),
            rx.button(
                "Next: Configure Training",
                on_click=rx.redirect("/configure"),
                size="3",
                margin_top="2em",
                color_scheme="blue",
                disabled=UploadState.file_path == "",
            ),
            align_items="center",
            padding="2em",
        )
    )
