import os

import torch

from src.module.model.unet import UNet
from src.pipeline import UNetDiffusionPipeline
from src.script.config import Config
from src.types_ import TimestepCallback


# TODO: add another parameter
def inference_unet(
    cfg: Config,
    timestep_callback: TimestepCallback | None = None,
) -> torch.Tensor:
    ci = cfg.infer
    cm = cfg.mel

    ckpt_path = ci.checkpoint_path

    if ckpt_path is None:
        raise ValueError("checkpoint_path must be specified")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    model = UNet.load_from_checkpoint(ckpt_path)

    pipeline = UNetDiffusionPipeline(model)

    # TODO: add pipeline parameters
    return pipeline(
        n_mels=cm.n_mels,
        length=cm.fixed_length,
        timestep_callback=timestep_callback,
    )
