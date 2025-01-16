"""
[descript-audio-codec/dac/nn/loss.py at main · descriptinc/descript-audio-codec](https://github.com/descriptinc/descript-audio-codec/blob/main/dac/nn/loss.py#L330)

そのまま拝借
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from vocoder.music_hifi.discriminator import Discriminator


class GANLoss(nn.Module):
    """
    Computes a discriminator loss, given a discriminator on
    generated waveforms/spectrograms compared to ground truth
    waveforms/spectrograms. Computes the loss for both the
    discriminator and the generator in separate functions.
    """

    def __init__(self, discriminator: Discriminator):
        super().__init__()
        self.discriminator = discriminator

    def forward(self, fake: torch.Tensor, real: torch.Tensor):
        d_fake = self.discriminator(fake)
        d_real = self.discriminator(real)
        return d_fake, d_real

    def discriminator_loss(
        self,
        fake: torch.Tensor,
        real: torch.Tensor,
    ) -> torch.Tensor:
        d_fake, d_real = self.forward(fake.clone().detach(), real)

        loss_d = sum(
            torch.mean(x_fake[-1] ** 2) + torch.mean((1 - x_real[-1]) ** 2)
            for x_fake, x_real in zip(d_fake, d_real)
        )
        return loss_d  # type: ignore

    def generator_loss(
        self,
        fake: torch.Tensor,
        real: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        d_fake, d_real = self.forward(fake, real)

        loss_g = sum(torch.mean((1 - x_fake[-1]) ** 2) for x_fake in d_fake)

        loss_feature = sum(
            F.l1_loss(d_fake[i][j], d_real[i][j].detach())
            for i in range(len(d_fake))
            for j in range(len(d_fake[i]) - 1)
        )
        return loss_g, loss_feature  # type: ignore
