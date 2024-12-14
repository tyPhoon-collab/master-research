from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.unet = UNet2DConditionModel()

        self.scheduler = DDPMScheduler()

    def forward(self, x, timesteps, context):
        return self.unet(x, timesteps, context).sample
