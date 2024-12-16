import os

import torch

from src.module.model.unet import UNet
from src.pipeline import UNetDiffusionPipeline
from src.script.config import InferConfig


# TODO: add genre parameter
# TODO: think about timestep_callback
def inference_unet(cfg: InferConfig) -> torch.Tensor:
    if cfg.checkpoint_path is None:
        raise ValueError("checkpoint_path must be specified")

    if not os.path.exists(cfg.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {cfg.checkpoint_path}")

    model = UNet.load_from_checkpoint(cfg.checkpoint_path)

    pipeline = UNetDiffusionPipeline(model)

    # TODO: add pipeline parameters
    return pipeline()
