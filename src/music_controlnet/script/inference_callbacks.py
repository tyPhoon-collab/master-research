import os
from abc import ABC, abstractmethod
from datetime import datetime
from logging import getLogger

import torch
from soundfile import write

from music_controlnet.plot import plot_spectrogram, plot_waveform

logger = getLogger(__name__)


class InferenceCallback(ABC):
    @abstractmethod
    def on_inference_start(self):
        pass

    @abstractmethod
    def on_inference_end(self, sample: torch.Tensor):
        pass

    @abstractmethod
    def on_timestep(self, timestep: int, sample: torch.Tensor):
        pass


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


class SaveOutputInferenceCallback(InferenceCallback):
    def __init__(self, save_dir: str, save_timesteps: list[int] | None = None) -> None:
        self.save_timesteps = save_timesteps or []
        self.save_dir = save_dir

        self._inference_start_datetime = None
        self._output_dir = None

    def on_inference_start(self):
        logger.info(f"Saving output at timesteps: {self.save_timesteps}")
        self._inference_start_datetime = datetime.now()
        self._output_dir = (
            f"{self.save_dir}/{self._inference_start_datetime:%Y%m%d_%H%M%S}"
        )
        os.makedirs(self._output_dir, exist_ok=True)

    def on_inference_end(self, sample: torch.Tensor):
        match sample.ndim:
            case 4:
                self._save_spectrogram(0, sample)
                pass
            case 3:
                self._save_waveform(0, sample)
                pass
            case _:
                raise ValueError(f"Unsupported shape: {sample.shape}")

    def on_timestep(self, timestep: int, sample: torch.Tensor):
        if timestep in self.save_timesteps:
            self._save_spectrogram(timestep, sample)

    def _save_spectrogram(self, timestep: int, sample: torch.Tensor):
        data = sample.squeeze().cpu().numpy()
        fig = plot_spectrogram(data)
        fig.write_image(f"{self._output_dir}/timestep_{timestep}_spectrogram.png")

    def _save_waveform(self, timestep: int, sample: torch.Tensor):
        data = sample.squeeze().cpu().numpy()
        fig = plot_waveform(data)
        fig.write_image(f"{self._output_dir}/timestep_{timestep}_waveform.png")
        write(f"{self._output_dir}/timestep_{timestep}.wav", data, 22050)
