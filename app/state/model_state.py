import reflex as rx


class ModelState(rx.State):
    model_name: str = "mistralai/Mistral-7B-v0.1"
    lora_r: list[int] = [16]
    lora_alpha: list[int] = [32]
    epochs: int = 3
    learning_rate: str = "2e-4"
    dataset_path: str = ""

    @rx.event
    def set_dataset_path(self, path: str):
        self.dataset_path = path

    @rx.event
    def set_model_name(self, name: str):
        self.model_name = name

    @rx.event
    def set_lora_r(self, value: list[float]):
        self.lora_r = [int(v) for v in value]

    @rx.event
    def set_lora_alpha(self, value: list[float]):
        self.lora_alpha = [int(v) for v in value]

    @rx.event
    def set_epochs(self, value: str):
        self.epochs = int(value)

    @rx.event
    def set_learning_rate(self, value: str):
        self.learning_rate = value

    @rx.event
    def start_training(self):
        """Stub — legacy pages redirect to /finetune for actual training."""
        return rx.redirect("/finetune")
