import os

import torch

from src.module.model.unet import UNet
from src.pipeline import UNetDiffusionPipeline
from src.script.config import InferConfig
from src.types_ import TimestepCallbackType


# TODO: add another parameter
def inference_unet(
    cfg: InferConfig,
    timestep_callback: TimestepCallbackType | None = None,
) -> torch.Tensor:
    if cfg.checkpoint_path is None:
        raise ValueError("checkpoint_path must be specified")

    if not os.path.exists(cfg.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {cfg.checkpoint_path}")

    model = UNet.load_from_checkpoint(cfg.checkpoint_path)

    pipeline = UNetDiffusionPipeline(model)

    # TODO: add pipeline parameters
    return pipeline(
        timestep_callback=timestep_callback,
    )
