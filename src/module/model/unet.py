import lightning as L
import torch
import torch.nn.functional as F
from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch.optim import Adam

from src.module.data.dataset import NUM_GENRES, PADDING_INDEX


class UNet(L.LightningModule):
    def __init__(self):
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

        self.lr = 1e-4

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

        loss = F.mse_loss(residual, noise)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)  # TODO: consider decay
        # scheduler = ...  # TODO: add scheduler
        return optimizer

    def forward(self, x, timestep, genres: torch.Tensor):
        return self.model(x, timestep, class_labels=genres)
