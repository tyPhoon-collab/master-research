from collections.abc import Callable
from typing import Any

import torch
from tqdm import tqdm

from src.module.model.unet import UNet

TimestepCallbackType = Callable[[int, torch.Tensor], None]


def _noop_timestep_callback(timestep: int, sample: torch.Tensor) -> None:
    pass


class UNetDiffusionPipeline:
    def __init__(self, model: UNet, scheduler: Any | None = None):
        self.model = model
        self.scheduler = scheduler

    @torch.inference_mode()
    def __call__(
        self,
        genres: torch.Tensor | None = None,
        timesteps: int = 1000,
        noise_shape: tuple = (1, 1, 160, 2560),
        timestep_callback: TimestepCallbackType | None = None,
        show_progress: bool = True,
    ) -> torch.Tensor:
        training = self.model.training

        try:
            self.model.eval()

            device = self.model.device

            # genresが指定されていない場合のデフォルト設定
            # Hip-Hop
            genres = genres or torch.tensor([[21]], device=device)

            scheduler = self.scheduler or self.model.scheduler
            scheduler.set_timesteps(timesteps)

            sample = torch.randn(noise_shape, device=device)

            callback = timestep_callback or _noop_timestep_callback

            tqdm_timesteps = tqdm(scheduler.timesteps, disable=not show_progress)

            for t in tqdm_timesteps:
                noise_pred = self.model(sample, t, genres).sample
                int_t = int(t.item())
                sample = scheduler.step(noise_pred, int_t, sample).prev_sample  # type: ignore

                callback(int_t, sample.clone().cpu())

            return sample

        finally:
            if training:
                self.model.train()
