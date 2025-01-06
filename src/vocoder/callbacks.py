import glob

import torchaudio
from lightning import LightningModule, Trainer

from callback.neptune import NeptuneLoggerCallback
from tool.config import MelConfig
from tool.pipeline import MelSpectrogramPipeline, WaveformPipeline
from tool.plot import fig_to_pil_image, plot_multiple, plot_spectrogram, plot_waveform
from vocoder.module.diffwave import DiffWaveLightning


class DiffWaveNeptuneLoggerCallback(NeptuneLoggerCallback):
    def __init__(self, test_dir: str, hop_length: int):
        super().__init__()

        self.test_dir = test_dir
        self.hop_length = hop_length

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        super().setup(trainer, pl_module, stage)

        c = MelConfig(
            hop_length=self.hop_length,
        )

        self.pipe_mel = MelSpectrogramPipeline(c).to(pl_module.device)
        self.pipe_waveform = WaveformPipeline(c).to(pl_module.device)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_train_epoch_end(trainer, pl_module)

        model: DiffWaveLightning = pl_module  # type: ignore

        for path in glob.glob(f"{self.test_dir}/*.mp3"):
            self._append(model, path)

    def _append(self, model: DiffWaveLightning, path: str):
        y, sr = torchaudio.load(path)
        y = y.to(model.device)

        waveform = self.pipe_waveform(y)
        mel = self.pipe_mel(y)

        waveform_hat = model.generate(
            mel,
            hop_length=self.hop_length,
        )
        mel_hat = self.pipe_mel(waveform_hat)

        fig = plot_multiple(
            [
                plot_waveform(waveform.squeeze().cpu().numpy()),
                plot_waveform(waveform_hat.squeeze().cpu().numpy()),
                plot_spectrogram(mel.squeeze().cpu().numpy()),
                plot_spectrogram(mel_hat.squeeze().cpu().numpy()),
            ]
        )
        img = fig_to_pil_image(fig)

        self.run["train/sample"].append(img)

        img.close()
