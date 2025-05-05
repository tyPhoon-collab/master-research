from typing import Any, Callable

import lightning as L
import torch
from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from tqdm import tqdm


class UNetLightning(L.LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        criterion: torch.nn.Module | None = None,
        num_class_embeds: int | None = None,
        clip_sample: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = UNet2DModel(
            in_channels=1,
            out_channels=1,
            num_class_embeds=num_class_embeds,
            block_out_channels=(64, 64, 128, 128, 256),
            up_block_types=(
                "ResnetUpsampleBlock2D",
                "ResnetUpsampleBlock2D",
                "ResnetUpsampleBlock2D",
                "ResnetUpsampleBlock2D",
                "ResnetUpsampleBlock2D",
            ),
            upsample_type="resnet",
            down_block_types=(
                "ResnetDownsampleBlock2D",
                "ResnetDownsampleBlock2D",
                "ResnetDownsampleBlock2D",
                "ResnetDownsampleBlock2D",
                "ResnetDownsampleBlock2D",
            ),
            downsample_type="resnet",
            layers_per_block=1,
        )

        self.scheduler = DDPMScheduler(
            clip_sample=clip_sample,
        )

        self.lr = lr
        self.criterion = criterion or torch.nn.L1Loss()

    def training_step(self, batch, batch_idx):
        mel = batch.get("spectrogram")
        genre = batch.get("genre") if self.model.config.num_class_embeds else None  # type: ignore

        assert mel is not None, "mel is None"

        noise = torch.randn_like(mel)
        timesteps = torch.randint(
            len(self.scheduler),
            (mel.size(0),),
            device=self.device,
        )
        noisy_mel = self.scheduler.add_noise(mel, noise, timesteps)  # type: ignore
        noise_pred = self(noisy_mel, timesteps, genre).sample
        loss = self.criterion(noise_pred, noise)

        self.log("train_loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        from schedulefree import RAdamScheduleFree

        self.optimizer = RAdamScheduleFree(self.parameters(), lr=self.lr)
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return self.optimizer

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        class_labels: torch.Tensor | None = None,
    ):
        return self.model(x, timestep, class_labels=class_labels)

    @torch.inference_mode()
    def generate(
        self,
        n_mels: int,
        length: int,
        genre: torch.Tensor | None = None,
        timesteps: int = 1000,
        on_timesteps: Callable[[int, torch.Tensor], None] | None = None,
        show_progress: bool = True,
        inference_scheduler: Any | None = None,
    ) -> torch.Tensor:
        training = self.training
        self.eval()

        noise_shape = (1, 1, n_mels, length)

        scheduler = inference_scheduler or self.scheduler
        scheduler.set_timesteps(timesteps)

        sample = torch.randn(noise_shape, device=self.device)

        on_timesteps = on_timesteps or (lambda t, x: None)

        for t in tqdm(
            scheduler.timesteps,
            desc="Generating",
            disable=not show_progress,
        ):
            noise_pred = self.model(sample, t, genre).sample
            sample = scheduler.step(noise_pred, int(t.item()), sample).prev_sample  # type: ignore

            on_timesteps(int(t.item()), sample)

        if training:
            self.train()

        return sample
