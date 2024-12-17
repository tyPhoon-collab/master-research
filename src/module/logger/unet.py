import io
from abc import ABC, abstractmethod

from neptune.types import File
from PIL import Image
from plotly.graph_objs._figure import Figure

from src.pipeline import UNetDiffusionPipeline
from src.plot import plot_spectrogram


class UNetLogger(ABC):
    @property
    def log(self):
        return self.model.log

    @property
    def logger(self):
        return self.model.logger

    def set_model(self, model):
        self.model = model

    @abstractmethod
    def training_step(self, loss, batch_idx: int): ...

    @abstractmethod
    def on_train_epoch_end(self) -> None: ...


class DefaultUNetLogger(UNetLogger):
    def training_step(self, loss, batch_idx: int):
        self.log("train_loss", loss, prog_bar=True)

    def on_train_epoch_end(self) -> None:
        pass


class NeptuneUNetLogger(DefaultUNetLogger):
    def __init__(self, **pipeline_kwargs):
        super().__init__()
        self._reset()

        self.pipeline_kwargs = pipeline_kwargs

    def set_model(self, model):
        super().set_model(model)

        self.pipeline = UNetDiffusionPipeline(self.model, self.model.scheduler)

    def training_step(self, loss, batch_idx: int):
        super().training_step(loss, batch_idx)

        self.logger.experiment[f"train/batch_{batch_idx}/loss"].append(loss)

        self.epoch_total_loss += loss.item()
        self.epoch_steps_count += 1

    def on_train_epoch_end(self) -> None:
        mean_epoch_loss = self.epoch_total_loss / self.epoch_steps_count
        self.logger.experiment["train/epoch_loss"].append(mean_epoch_loss)
        self._reset()

        sample = self.pipeline(**self.pipeline_kwargs)
        data = sample[0][0].cpu().numpy()

        fig = plot_spectrogram(data)
        file = self._to_file(fig)

        self.logger.experiment["train/sample"].append(file)

    def _to_file(self, fig: Figure):
        bytes = fig.to_image("png")
        buf = io.BytesIO(bytes)
        img = Image.open(buf)

        file = File.as_image(img)

        img.close()

        return file

    def _reset(self):
        self.epoch_total_loss = 0.0
        self.epoch_steps_count = 0
