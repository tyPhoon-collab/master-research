import torch

from src.module.model.unet import UNet
from src.pipeline import UNetDiffusionPipeline


# TODO: add genre parameter
# TODO: think about timestep_callback
def inference_unet(
    checkpoint_path: str,
    **kwargs,
) -> torch.Tensor:
    model = UNet.load_from_checkpoint(checkpoint_path)

    pipeline = UNetDiffusionPipeline(model)

    return pipeline(**kwargs)
