import os
from logging import getLogger

import torch

from tool.plot import plot_spectrogram

logger = getLogger(__name__)


class InferenceCallback:
    def on_inference_start(self):
        pass

    def on_inference_end(self, sample: torch.Tensor):
        pass

    def on_timestep(self, timestep: int, sample: torch.Tensor):
        pass


class StoreDirectory:
    def set_dir(self, dir: str):
        self.dir = dir

    def make_dir(self):
        assert (
            self.dir is not None
        ), "Directory is not set. Please call set_dir() first."
        os.makedirs(self.dir, exist_ok=True)
        return self.dir


class ComposeInferenceCallback(InferenceCallback):
    def __init__(self, callbacks: list[InferenceCallback] | None = None) -> None:
        super().__init__()
        self.callbacks = callbacks or []

    def on_inference_start(self):
        for callback in self.callbacks:
            callback.on_inference_start()

    def on_inference_end(self, sample: torch.Tensor):
        for callback in self.callbacks:
            callback.on_inference_end(sample)

    def on_timestep(self, timestep: int, sample: torch.Tensor):
        for callback in self.callbacks:
            callback.on_timestep(timestep, sample)


class SaveInferenceCallback(InferenceCallback, StoreDirectory):
    def __init__(
        self,
        save_timesteps: list[int] | None = None,
    ) -> None:
        self.save_timesteps = save_timesteps or []

    def on_inference_start(self):
        logger.info(f"Saving output at timesteps: {self.save_timesteps}")
        self._save_dir = self.make_dir()

    def on_inference_end(self, sample: torch.Tensor):
        self._save_spectrogram(0, sample)

    def on_timestep(self, timestep: int, sample: torch.Tensor):
        if timestep in self.save_timesteps:
            self._save_spectrogram(timestep, sample)

    def _save_spectrogram(self, timestep: int, sample: torch.Tensor):
        data = sample.squeeze().cpu().numpy()
        fig = plot_spectrogram(data)
        fig.write_image(f"{self._save_dir}/timestep_{timestep}_spectrogram.png")
