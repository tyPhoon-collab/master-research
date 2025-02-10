from typing import Any

from lightning import LightningModule, Trainer

from callback.neptune import NeptuneLoggerCallback
from tool.plot import fig_to_pil_image, plot_multiple, plot_spectrogram, plot_waveform
from vocoder.module.diffwave import DiffWaveLightning


class DiffWaveNeptuneLoggerCallback(NeptuneLoggerCallback):
    def __init__(self, hop_length: int) -> None:
        super().__init__()
        self.hop_length = hop_length

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

        fig = plot_multiple(
            [
                plot_waveform(waveform.squeeze().cpu().numpy()),
                plot_waveform(waveform_hat.squeeze().cpu().numpy()),
                plot_spectrogram(mel.squeeze().cpu().numpy()),
            ]
        )
        img = fig_to_pil_image(fig)

        self.run["train/sample"].append(img)

        img.close()
