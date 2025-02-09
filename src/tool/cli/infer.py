import os

import torch

from music_controlnet.module.unet import UNetLightning
from tool.config import Config
from vocoder.module.diffwave import DiffWaveLightning


class UNetGenerator:
    def __init__(self, ckpt_path: str, n_mels: int, length: int):
        self.ckpt_path = _check_ckpt_path(ckpt_path)

        self.model = UNetLightning.load_from_checkpoint(self.ckpt_path)
        # self.n_mels = self.model.n_mels
        self.n_mels = n_mels
        self.length = length

    def __call__(self):
        mel = self.model.generate(
            n_mels=self.n_mels,
            length=self.length,
        )
        return mel


class DiffWaveVocoder:
    def __init__(self, ckpt_path: str, n_mels: int, hop_length: int) -> None:
        self.ckpt_path = _check_ckpt_path(ckpt_path)
        self.model = DiffWaveLightning.load_from_checkpoint(
            ckpt_path,
            n_mels=n_mels,
        )

        self.hop_length = hop_length

    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        waveform = self.model.generate(
            mel,
            hop_length=self.hop_length,
        )

        return waveform


def inference(cfg: Config) -> torch.Tensor:
    gen = cfg.infer.generator_object
    voc = cfg.infer.vocoder_object

    mel = gen()
    waveform = voc(mel)

    _save_spectrogram(cfg.infer.save_dir, mel)
    _save_waveform(cfg.infer.save_dir, waveform)

    return waveform


def _check_ckpt_path(ckpt_path: str | None) -> str:
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
