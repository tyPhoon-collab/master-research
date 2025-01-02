import os

import torch
from hydra.utils import instantiate

from music_controlnet.module.unet import PostPipeline, UNetLightning
from tool.config import Config
from tool.pipeline import (
    InverseMelSpectrogramPipeline,
)


def inference_unet(
    cfg: Config,
    post_pipeline: PostPipeline | None = None,
) -> torch.Tensor:
    ci = cfg.infer
    cm = cfg.mel

    ckpt_path = _check_ckpt_path(ci)

    model = UNetLightning.load_from_checkpoint(ckpt_path)

    callbacks = (
        [instantiate(callback) for callback in ci.callbacks] if ci.callbacks else None
    )

    return model.generate(
        n_mels=cm.n_mels,
        length=cm.fixed_length,
        callbacks=callbacks,
        post_pipeline=post_pipeline,
    )


def _check_ckpt_path(ci):
    ckpt_path = ci.checkpoint_path

    if ckpt_path is None:
        raise ValueError("checkpoint_path must be specified")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
    return ckpt_path


def inference(cfg: Config):
    pipeline = InverseMelSpectrogramPipeline(cfg.mel)
    data = inference_unet(cfg, post_pipeline=pipeline)
    return data
