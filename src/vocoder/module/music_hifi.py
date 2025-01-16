import lightning as L
import torch
from omegaconf import DictConfig

from vocoder.music_hifi.discriminator import Discriminator
from vocoder.music_hifi.generator import Generator
from vocoder.music_hifi.loss import GANLoss


class MusicHiFiLightning(L.LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        sampling_rate: int = 22050,
        n_mels: int = 128,
    ):
        super().__init__()
        self.generator = Generator(
            DictConfig(
                {
                    "num_mels": n_mels,
                    #
                    # "upsample_rates": [8, 4, 2, 2, 2, 2],
                    # "upsample_kernel_sizes": [16, 8, 4, 4, 4, 4],
                    # "upsample_initial_channel": 1536,
                    # "resblock_kernel_sizes": [3, 7, 11],
                    # "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                    "upsample_rates": [8, 8, 2, 2],
                    "upsample_kernel_sizes": [16, 16, 4, 4],
                    "upsample_initial_channel": 512,
                    "resblock_kernel_sizes": [3, 7, 11],
                    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                    #
                    "use_tanh_at_final": False,
                    "use_bias_at_final": False,
                }
            )
        )
        self.discriminator = Discriminator(
            sample_rate=sampling_rate,
        )  # TODO: check params from paper

        self.gan_loss = GANLoss(self.discriminator)

        self.lr = lr

        self.automatic_optimization = False

    def forward(self, x: torch.Tensor):
        return self.generator(x)

    def configure_optimizers(self):
        from schedulefree import RAdamScheduleFree

        optimizer_g = RAdamScheduleFree(self.generator.parameters(), lr=self.lr)
        optimizer_d = RAdamScheduleFree(self.discriminator.parameters(), lr=self.lr)

        return [optimizer_g, optimizer_d]

    def training_step(self, batch):
        optimizer_g, optimizer_d = self.optimizers()  # type: ignore

        waveform = batch["waveform"]
        mel = batch["mel"].squeeze(1)

        waveform_hat: torch.Tensor = self.generator(mel)

        # Discriminator
        loss_d = self.gan_loss.discriminator_loss(
            waveform_hat.detach(),
            waveform,
        )

        optimizer_d.zero_grad()
        self.manual_backward(loss_d)
        optimizer_d.step()

        self.log("train_d_loss", loss_d, prog_bar=True)

        # Generator
        loss_g, loss_feature = self.gan_loss.generator_loss(
            waveform_hat,
            waveform,
        )
        total_loss_g = loss_g + loss_feature

        optimizer_g.zero_grad()
        self.manual_backward(total_loss_g)
        optimizer_g.step()

        self.log("train_g_loss", loss_g, prog_bar=True)
        self.log("train_feature_loss", loss_feature, prog_bar=True)
        self.log("train_loss", total_loss_g, prog_bar=True)
