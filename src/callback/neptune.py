from collections.abc import Mapping
from typing import Any

from lightning import Callback, LightningModule, Trainer
from neptune import Run
from torch import Tensor


class NeptuneLoggerCallback(Callback):
    def __init__(self):
        super().__init__()

        self.epoch_total_loss = 0.0
        self.epoch_steps_count = 0

    def _reset(self):
        self.epoch_total_loss = 0.0
        self.epoch_steps_count = 0

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.run: Run = pl_module.logger.experiment  # type: ignore

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
    ) -> None:
        loss = outputs["loss"]  # type: ignore
        self.run[f"train/batch_{pl_module.current_epoch}/loss"].append(loss)

        self.epoch_total_loss += loss.item()
        self.epoch_steps_count += 1

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        mean_epoch_loss = self.epoch_total_loss / self.epoch_steps_count
        self.run["train/epoch_loss"].append(mean_epoch_loss)
        self._reset()
