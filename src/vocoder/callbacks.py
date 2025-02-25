from collections.abc import Callable
from typing import Any

import torch
from lightning import LightningModule, Trainer

from callback.neptune import NeptuneLoggerCallback
from tool.plot import fig_to_pil_image, plot_multiple, plot_spectrogram, plot_waveform
from vocoder.module.diffwave import DiffWaveLightning


class DiffWaveNeptuneLoggerCallback(NeptuneLoggerCallback):
    def __init__(
        self,
        hop_length: int,
        mel_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.hop_length = hop_length
        self.mel_transform = mel_transform

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_fit_start(trainer, pl_module)

        if self.mel_transform is not None:
            op = getattr(self.mel_transform, "to", None)
            if callable(op):
                self.mel_transform = self.mel_transform.to(pl_module.device)  # type: ignore

    def on_validation_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        model: DiffWaveLightning = pl_module  # type: ignore
        self._append(model, batch)

    def _append(self, model: DiffWaveLightning, batch: Any) -> None:
        waveform = batch["waveform"]
        mel = batch["mel"]

        waveform_hat = model.generate(
            mel,
            hop_length=self.hop_length,
        )

        figs = [
            plot_waveform(waveform.squeeze().cpu().numpy()),
            plot_waveform(waveform_hat.squeeze().cpu().numpy()),
            plot_spectrogram(mel.squeeze().cpu().numpy()),
        ]

        if self.mel_transform is not None:
            mel_hat = self.mel_transform(waveform_hat)
            figs.append(plot_spectrogram(mel_hat.squeeze().cpu().numpy()))

        fig = plot_multiple(figs)
        img = fig_to_pil_image(fig)

        self.run["train/sample"].append(img)

        img.close()
