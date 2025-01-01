import requests
from lightning import Callback, LightningModule, Trainer


class DiscordCallback(Callback):
    def __init__(self, webhook_url: str) -> None:
        self.webhook_url = webhook_url

    def _send_message(self, message: str) -> None:
        requests.post(self.webhook_url, json={"content": message})

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._send_message("Training finished")

    def on_exception(
        self, trainer: Trainer, pl_module: LightningModule, exception: BaseException
    ) -> None:
        self._send_message(f"Exception occurred: {exception}")
