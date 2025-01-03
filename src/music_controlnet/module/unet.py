from collections.abc import Callable
from typing import Any, Dict, Literal

import lightning as L
import torch
from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from tqdm import tqdm

from fma.metadata import NUM_GENRES, PADDING_INDEX
from music_controlnet.module.inference_callbacks import (
    ComposeInferenceCallback,
    InferenceCallback,
)

PostPipeline = Callable[[torch.Tensor], torch.Tensor] | torch.nn.Module


class UNetLightning(L.LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        criterion: torch.nn.Module | None = None,
    ):
        super().__init__()

        num_class_embeds = NUM_GENRES

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

        # TODO: Is an EmbeddingBag good?
        self.model.class_embedding = torch.nn.EmbeddingBag(  # type: ignore
            num_class_embeds,
            256,
            padding_idx=PADDING_INDEX,
        )
        self.scheduler = DDPMScheduler()

        self.lr = lr
        self.criterion = criterion or torch.nn.L1Loss()

    def training_step(self, batch, batch_idx):
        mel = batch["mel"]
        genres = batch["genres"]

        noise = torch.randn_like(mel)
        timesteps = torch.randint(
            len(self.scheduler),
            (mel.size(0),),
            device=self.device,
        )
        noisy_mel = self.scheduler.add_noise(mel, noise, timesteps)  # type: ignore
        noise_pred = self(noisy_mel, timesteps, genres).sample
        loss = self.criterion(noise_pred, noise)

        self.log("train_loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        from schedulefree import RAdamScheduleFree

        self.optimizer = RAdamScheduleFree(self.parameters(), lr=self.lr)
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return self.optimizer

    # optimizer hooks
    def on_train_epoch_start(self):
        self._set_optimizer_mode("train")

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self._set_optimizer_mode("eval")

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self._set_optimizer_mode("train")

    def _set_optimizer_mode(self, mode: Literal["train", "eval"]) -> None:
        # ScheduleFreeなOptimizerはtrain/evalを切り替える必要がある場合がある
        if hasattr(self, "optimizer") and callable(getattr(self.optimizer, mode, None)):
            getattr(self.optimizer, mode)()

    # end optimizer hooks

    def forward(self, x: torch.Tensor, timestep: torch.Tensor, genres: torch.Tensor):
        return self.model(x, timestep, class_labels=genres)

    @torch.inference_mode()
    def generate(
        self,
        n_mels: int,
        length: int,
        genres: torch.Tensor | None = None,
        timesteps: int = 1000,
        callbacks: list[InferenceCallback] | None = None,
        show_progress: bool = True,
        inference_scheduler: Any | None = None,
        post_pipeline: PostPipeline | None = None,
    ) -> torch.Tensor:
        training = self.training
        self.eval()

        noise_shape = (1, 1, n_mels, length)

        callback = ComposeInferenceCallback(callbacks)

        scheduler = inference_scheduler or self.scheduler
        scheduler.set_timesteps(timesteps)

        if isinstance(post_pipeline, torch.nn.Module):
            post_pipeline = post_pipeline.to(self.device)

        callback.on_inference_start()

        # Hip-Hop: 21
        genres = genres or torch.tensor([[21]], device=self.device)
        sample = torch.randn(noise_shape, device=self.device)

        for t in tqdm(
            scheduler.timesteps,
            desc="Generating",
            disable=not show_progress,
        ):
            noise_pred = self.model(sample, t, genres).sample
            sample = scheduler.step(noise_pred, int(t.item()), sample).prev_sample  # type: ignore

            callback.on_timestep(int(t.item()), sample)

        if post_pipeline:
            sample = post_pipeline(sample)

        callback.on_inference_end(sample)

        if training:
            self.train()

        return sample
