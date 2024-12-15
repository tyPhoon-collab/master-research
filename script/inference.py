import torch

from src.module.model.unet import UNet


# TODO: add genre parameter
def inference_unet(
    checkpoint_path: str, genres: torch.Tensor | None = None
) -> torch.Tensor:
    model = UNet.load_from_checkpoint(checkpoint_path)

    # TODO: consider genres
    # とりあえずHip-Hopのジャンルを指定
    genres = genres or torch.tensor([21])

    model.eval()
    # TODO: consider custom scheduler
    scheduler = model.scheduler
    scheduler.set_timesteps(1000)

    # TODO: consider noise shape
    sample = torch.randn(1, 1, 160, 2560)

    for t in scheduler.timesteps:
        with torch.no_grad():
            noise_pred = model(t, genres).sample
        sample = scheduler.step(noise_pred, int(t.item()), sample).prev_sample  # type: ignore

    return sample
