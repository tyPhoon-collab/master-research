import os
from collections.abc import Callable

import torch
from hydra.utils import instantiate

from src.module.model.unet import UNet
from src.pipeline import InverseMelSpectrogramPipeline, UNetDiffusionPipeline
from src.script.config import Config
from src.utils import auto_device


# TODO: add another parameter
def inference_unet(
    cfg: Config,
    post_pipeline: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    ci = cfg.infer
    cm = cfg.mel

    ckpt_path = _check_ckpt_path(ci)

    model = UNet.load_from_checkpoint(ckpt_path)

    pipeline = UNetDiffusionPipeline(
        model,
        post_pipeline=post_pipeline,
    )
    callbacks = (
        [instantiate(callback) for callback in ci.callbacks] if ci.callbacks else None
    )

    # TODO: add pipeline parameters
    return pipeline(n_mels=cm.n_mels, length=cm.fixed_length, callbacks=callbacks)


def _check_ckpt_path(ci):
    ckpt_path = ci.checkpoint_path

    if ckpt_path is None:
        raise ValueError("checkpoint_path must be specified")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
    return ckpt_path


def inference(cfg: Config):
    device = auto_device()
    pipeline = InverseMelSpectrogramPipeline(cfg.mel).to(device)
    data = inference_unet(cfg, post_pipeline=pipeline)
    return data
