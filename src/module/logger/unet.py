from abc import ABC, abstractmethod

import lightning as L
from neptune import Run

from src.pipeline import UNetDiffusionPipeline
from src.plot import plot_mel_spectrogram_by_librosa
from src.script.config import Config


class ModelLogger(ABC):
    @property
    def log(self):
        return self.model.log

    @property
    def logger(self):
        return self.model.logger

    def set_model(self, model):
        self.model: L.LightningModule = model

    def set_config(self, config: Config):
        self.config = config

    @abstractmethod
    def training_step(self, loss, batch_idx: int): ...

    @abstractmethod
    def on_train_epoch_end(self) -> None: ...


class SimpleModelLogger(ModelLogger):
    def training_step(self, loss, batch_idx: int):
        self.log("train_loss", loss, prog_bar=True)

    def on_train_epoch_end(self) -> None:
        pass


class NeptuneUNetLogger(SimpleModelLogger):
    def __init__(self, timesteps: int = 1000):
        super().__init__()
        self._reset()

        self.timesteps = timesteps

    @property
    def run(self) -> Run:
        return self.logger.experiment  # type: ignore

    def set_model(self, model: L.LightningModule):
        super().set_model(model)

        self.pipeline = UNetDiffusionPipeline(self.model, self.model.scheduler)

    def training_step(self, loss, batch_idx: int):
        super().training_step(loss, batch_idx)

        self.run[f"train/batch_{self.model.current_epoch}/loss"].append(loss)

        self.epoch_total_loss += loss.item()
        self.epoch_steps_count += 1

    def on_train_epoch_end(self) -> None:
        mean_epoch_loss = self.epoch_total_loss / self.epoch_steps_count
        self.run["train/epoch_loss"].append(mean_epoch_loss)
        self._reset()

        sample = self.pipeline(
            n_mels=self.config.mel.n_mels,
            length=self.config.mel.fixed_length,
            timesteps=self.timesteps,
        )
        data = sample[0][0].cpu().numpy()

        fig = plot_mel_spectrogram_by_librosa(data)

        self.run["train/sample"].append(fig)

    def _reset(self):
        self.epoch_total_loss = 0.0
        self.epoch_steps_count = 0
