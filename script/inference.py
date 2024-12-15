from collections.abc import Callable

import numpy as np
import torch

from src.module.model.unet import UNet


# TODO: add genre parameter
# TODO: think about timestep_callback
def inference_unet(
    checkpoint_path: str,
    *,
    genres: torch.Tensor | None = None,
    timestep_callback: Callable[[int, np.ndarray], None] | None = None,
) -> torch.Tensor:
    model = UNet.load_from_checkpoint(checkpoint_path)

    # TODO: consider genres
    # とりあえずHip-Hopのジャンルを指定
    genres = genres or torch.tensor([[21]], device=model.device)

    model.eval()
    # TODO: consider custom scheduler
    scheduler = model.scheduler
    scheduler.set_timesteps(1000)

    # TODO: consider noise shape
    sample = torch.randn(1, 1, 160, 2560, device=model.device)

    callback = timestep_callback or (lambda t, s: None)

    for t in scheduler.timesteps:
        with torch.no_grad():
            noise_pred = model(sample, t, genres).sample

            int_t = int(t.item())
            sample = scheduler.step(noise_pred, int_t, sample).prev_sample  # type: ignore
            callback(int_t, sample.clone().cpu().numpy())

    return sample
