from lightning import LightningModule, Trainer

from callback.neptune import NeptuneLoggerCallback
from music_controlnet.module.unet import UNetLightning
from visualize.plot import fig_to_pil_image, plot_spectrogram


class UNetNeptuneLoggerCallback(NeptuneLoggerCallback):
    def __init__(self, timesteps: int = 1000, n_mels: int = 80, length: int = 2560):
        super().__init__()
        self.timesteps = timesteps

        self.n_mels = n_mels
        self.length = length

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_train_epoch_end(trainer, pl_module)

        model: UNetLightning = pl_module  # type: ignore

        sample = model.generate(
            n_mels=self.n_mels,
            length=self.length,
            timesteps=self.timesteps,
        )
        data = sample[0][0].cpu().numpy()

        fig = plot_spectrogram(data)
        img = fig_to_pil_image(fig)

        self.run["train/sample"].append(img)

        img.close()
