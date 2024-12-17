import requests
from lightning import Callback, LightningModule, Trainer


class DiscordCallback(Callback):
    def __init__(self, webhook_url: str) -> None:
        self.webhook_url = webhook_url

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_train_end(trainer, pl_module)

        self._sendMessage("Training finished")

    def _sendMessage(self, message: str) -> None:
        requests.post(self.webhook_url, json={"content": message})
