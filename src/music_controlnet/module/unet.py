from collections.abc import Callable
from typing import Any

import lightning as L
import torch
from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch.optim import Adam
from tqdm import tqdm

from fma.metadata import NUM_GENRES, PADDING_INDEX
from music_controlnet.module.inference_callbacks import (
    ComposeInferenceCallback,
    InferenceCallback,
)

PostPipeline = Callable[[torch.Tensor], torch.Tensor] | torch.nn.Module


class UNet(L.LightningModule):
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
        mel, genres = batch
        noise = torch.randn_like(mel)
        timesteps = torch.randint(
            len(self.scheduler),
            (mel.size(0),),
            device=self.device,
        )
        noisy_mel = self.scheduler.add_noise(mel, noise, timesteps)  # type: ignore

        # forward
        residual = self(noisy_mel, timesteps, genres).sample

        # ノイズとの損失を計算。self.modelはノイズを出力する
        loss = self.criterion(residual, noise)

        self.log("train_loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)  # TODO: consider decay
        return optimizer

    def forward(self, x, timestep, genres: torch.Tensor):
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
        scheduler: Any | None = None,
        post_pipeline: PostPipeline | None = None,
    ) -> torch.Tensor:
        training = self.model.training
        noise_shape = (1, 1, n_mels, length)
        callback = ComposeInferenceCallback(callbacks)
        scheduler_ = scheduler or self.scheduler
        post_pipeline_ = (
            post_pipeline.to(self.device)
            if isinstance(post_pipeline, torch.nn.Module)
            else post_pipeline
        )

        try:
            self.model.eval()

            callback.on_inference_start()

            device = self.model.device

            # genresが指定されていない場合のデフォルト設定
            # Hip-Hop
            genres = genres or torch.tensor([[21]], device=device)

            scheduler = self.scheduler or self.model.scheduler
            scheduler_.set_timesteps(timesteps)

            sample = torch.randn(noise_shape, device=device)

            tqdm_timesteps = tqdm(
                scheduler_.timesteps,
                desc="Generating",
                disable=not show_progress,
            )

            for t in tqdm_timesteps:
                noise_pred = self.model(sample, t, genres).sample
                int_t = int(t.item())
                sample = scheduler.step(noise_pred, int_t, sample).prev_sample  # type: ignore

                callback.on_timestep(int_t, sample)

            if post_pipeline_:
                sample = post_pipeline_(sample)

            callback.on_inference_end(sample)
            return sample

        finally:
            if training:
                self.model.train()
