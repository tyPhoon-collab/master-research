from abc import ABC, abstractmethod

import lightning as L
from neptune import Run

from music_controlnet.plot import plot_spectrogram, plotly_fig_to_pil_image
from music_controlnet.script.config import Config


class ModelLogger(ABC):
    @property
    def logger(self):
        return self.model.logger

    def set_model(self, model):
        self.model: L.LightningModule = model

    def set_config(self, config: Config):
        self.config = config

    @abstractmethod
    def on_batch_loss(self, loss):
        pass

    @abstractmethod
    def on_epoch_end(self):
        pass


class NeptuneUNetLogger(ModelLogger):
    def __init__(self, timesteps: int = 1000):
        super().__init__()
        self._reset()

        self.timesteps = timesteps

    @property
    def run(self) -> Run:
        return self.logger.experiment  # type: ignore

    def set_model(self, model: L.LightningModule):
        super().set_model(model)

    def on_batch_loss(self, loss):
        self.run[f"train/batch_{self.model.current_epoch}/loss"].append(loss)

        self.epoch_total_loss += loss.item()
        self.epoch_steps_count += 1

    def on_epoch_end(self) -> None:
        mean_epoch_loss = self.epoch_total_loss / self.epoch_steps_count
        self.run["train/epoch_loss"].append(mean_epoch_loss)
        self._reset()

        sample = self.model.generate(
            n_mels=self.config.mel.n_mels,
            length=self.config.mel.fixed_length,
            timesteps=self.timesteps,
        )
        data = sample[0][0].cpu().numpy()

        fig = plot_spectrogram(data)
        img = plotly_fig_to_pil_image(fig)

        self.run["train/sample"].append(img)

        img.close()

    def _reset(self):
        self.epoch_total_loss = 0.0
        self.epoch_steps_count = 0
