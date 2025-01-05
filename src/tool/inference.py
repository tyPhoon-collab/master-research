import os

import numpy as np
import torch

from music_controlnet.module.unet import UNetLightning
from tool.config import Config
from tool.pipeline import InverseMelSpectrogramPipeline, MelSpectrogramPipeline
from tool.plot import plot_waveform
from vocoder.module.diffwave import DiffWaveLightning


def inference_unet(cfg: Config) -> torch.Tensor:
    ci = cfg.infer
    cm = cfg.mel

    ckpt_path = _check_ckpt_path(ci)
    model = UNetLightning.load_from_checkpoint(ckpt_path)

    return model.generate(
        n_mels=cm.n_mels,
        length=cm.fixed_length,
        callbacks=ci.callbacks_objects,
    )


def inference_diffwave(cfg: Config) -> torch.Tensor:
    ci = cfg.infer
    cm = cfg.mel

    ckpt_path = _check_ckpt_path(ci)
    model = DiffWaveLightning.load_from_checkpoint(ckpt_path)

    pipe = MelSpectrogramPipeline(cm).to(model.device)
    mel = pipe("data/audio/000010.mp3")

    waveform = model.generate(
        mel,
        hop_length=cm.hop_length,
    )

    return waveform


def inference(cfg: Config) -> torch.Tensor:
    mel = inference_unet(cfg)
    pipe = InverseMelSpectrogramPipeline(cfg.mel).to("cuda")

    waveform = pipe(mel)
    _save_waveform(
        waveform.cpu().squeeze().numpy(),
        cfg.infer.save_dir,
    )

    return waveform


def _check_ckpt_path(ci):
    ckpt_path = ci.checkpoint_path

    if ckpt_path is None:
        raise ValueError("checkpoint_path must be specified")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
    return ckpt_path


def _save_waveform(data: np.ndarray, save_dir: str):
    from soundfile import write

    fig = plot_waveform(data)
    fig.write_image(f"{save_dir}/waveform.png")
    write(f"{save_dir}/output.wav", data, 22050)
