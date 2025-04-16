from typing import Any

import lightning as L
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffwave.model import DiffWave
from omegaconf import DictConfig
from tqdm import tqdm


class DiffWaveLightning(L.LightningModule):
    def __init__(
        self,
        n_mels: int,
        lr: float = 2e-4,
        criterion: torch.nn.Module | None = None,
        num_train_timesteps: int = 50,
        residual_layers: int = 30,
        residual_channels: int = 64,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=1e-4,
            beta_end=0.05,
        )
        self.lr = lr
        self.criterion = criterion or torch.nn.L1Loss()

        self.model = DiffWave(
            DictConfig(
                {
                    "residual_layers": residual_layers,
                    "residual_channels": residual_channels,
                    "dilation_cycle_length": 10,
                    "noise_schedule": [t.item() for t in self.scheduler.betas],
                    "n_mels": n_mels,
                }
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        spectrogram: torch.Tensor,
    ):
        # drop channel dimension
        return self.model(x.squeeze(1), spectrogram.squeeze(1), timestep)

    def training_step(self, batch, batch_idx):
        waveform = batch["waveform"]
        spectrogram = batch["spectrogram"]

        noise = torch.randn_like(waveform)
        timestep = torch.randint(
            len(self.scheduler),
            (waveform.size(0),),
            device=self.device,
        )
        noisy_waveform = self.scheduler.add_noise(waveform, noise, timestep)  # type: ignore
        noise_pred = self(noisy_waveform, timestep, spectrogram)
        loss = self.criterion(noise_pred, noise)

        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Callbackが適切に呼ばれるのために、空のvalidation_stepを追加
        """
        pass

    def configure_optimizers(self):
        from schedulefree import RAdamScheduleFree

        self.optimizer = RAdamScheduleFree(self.parameters(), lr=self.lr)
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return self.optimizer

    @torch.inference_mode()
    def generate(
        self,
        spectrogram: torch.Tensor,
        hop_length: int = 256,
        inference_scheduler: Any | None = None,
        show_progress: bool = True,
    ):
        training = self.training
        self.eval()

        scheduler = inference_scheduler or self.scheduler

        length = spectrogram.size(-1) * hop_length
        noise_shape = (1, 1, length)
        waveform = torch.randn(noise_shape, device=self.device)

        for t in tqdm(
            scheduler.timesteps,
            desc="Generating",
            disable=not show_progress,
        ):
            noise_pred = self(waveform, t, spectrogram)
            waveform = scheduler.step(noise_pred, int(t.item()), waveform).prev_sample  # type: ignore

        if training:
            self.train()

        return waveform
