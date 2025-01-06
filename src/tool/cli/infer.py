import os

import torch
import torchaudio

from tool.config import Config
from tool.pipeline import InverseMelSpectrogramPipeline, MelSpectrogramPipeline


def inference_unet(cfg: Config) -> torch.Tensor:
    from music_controlnet.module.unet import UNetLightning

    ci = cfg.infer
    cm = cfg.mel

    ckpt_path = _check_ckpt_path(ci)
    model = UNetLightning.load_from_checkpoint(ckpt_path)

    mel = model.generate(
        n_mels=cm.n_mels,
        length=cm.fixed_length,
    )

    _save_spectrogram(ci.save_dir, mel)

    return mel


def inference_diffwave(cfg: Config) -> torch.Tensor:
    from vocoder.module.diffwave import DiffWaveLightning

    ci = cfg.infer
    cm = cfg.mel

    ckpt_path = _check_ckpt_path(ci)
    model = DiffWaveLightning.load_from_checkpoint(
        ckpt_path,
        n_mels=cm.n_mels,
    )

    pipe = MelSpectrogramPipeline(cm).to(model.device)
    y, _ = torchaudio.load("data/audio/000010.mp3")
    y = y.to(model.device)
    mel = pipe(y)

    waveform = model.generate(
        mel,
        hop_length=cm.hop_length,
    )

    _save_waveform(ci.save_dir, waveform)

    return waveform


def inference(cfg: Config) -> torch.Tensor:
    mel = inference_unet(cfg)
    pipe = InverseMelSpectrogramPipeline(cfg.mel).to("cuda")

    waveform = pipe(mel)
    _save_waveform(cfg.infer.save_dir, waveform)

    return waveform


def _check_ckpt_path(ci):
    ckpt_path = ci.checkpoint_path

    if ckpt_path is None:
        raise ValueError("checkpoint_path must be specified")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
    return ckpt_path


def _save_waveform(save_dir: str, waveform: torch.Tensor):
    from soundfile import write

    from tool.plot import plot_waveform

    os.makedirs(save_dir, exist_ok=True)

    data = waveform.squeeze().cpu().numpy()

    fig = plot_waveform(data)
    fig.write_image(f"{save_dir}/waveform.png")
    write(f"{save_dir}/output.wav", data, 22050)


def _save_spectrogram(save_dir: str, mel: torch.Tensor):
    from tool.plot import plot_spectrogram

    os.makedirs(save_dir, exist_ok=True)

    data = mel.squeeze().cpu().numpy()
    fig = plot_spectrogram(data)
    fig.write_image(f"{save_dir}/spectrogram.png")
