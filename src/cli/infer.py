import os

import torch

from bigvgan import BigVGAN
from cli.config import Config
from music_controlnet.module.unet import UNetLightning
from vocoder.module.diffwave import DiffWaveLightning


class UNetGenerator:
    def __init__(
        self, ckpt_path: str, n_mels: int, length: int, timesteps: int = 1000
    ) -> None:
        self.ckpt_path = _check_ckpt_path(ckpt_path)

        self.model = UNetLightning.load_from_checkpoint(self.ckpt_path)
        # self.n_mels = self.model.n_mels
        self.n_mels = n_mels
        self.length = length
        self.timesteps = timesteps

    def __call__(self):
        mel = self.model.generate(
            n_mels=self.n_mels,
            length=self.length,
            timesteps=self.timesteps,
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


class BigVGANVocoder:
    def __init__(self) -> None:
        model = BigVGAN.from_pretrained(
            "nvidia/bigvgan_v2_44khz_128band_256x", use_cuda_kernel=False
        )

        # remove weight norm in the model and set to eval
        model.remove_weight_norm()
        model = model.eval().to("cpu")

        self.model = model

    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        # generate waveform from mel
        with torch.inference_mode():
            wav_gen = self.model(
                mel
            )  # wav_gen is FloatTensor with shape [B(1), 1, T_time] and values in [-1, 1]
        return wav_gen


def inference(cfg: Config) -> torch.Tensor:
    gen = cfg.infer.generator_object
    voc = cfg.infer.vocoder_object

    mel = gen()
    waveform = voc(mel.squeeze(0))

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

    from visualize.plot import plot_waveform

    os.makedirs(save_dir, exist_ok=True)

    data = waveform.squeeze().cpu().numpy()

    fig = plot_waveform(data)
    fig.write_image(f"{save_dir}/waveform.png")
    write(f"{save_dir}/output.wav", data, 22050)


def _save_spectrogram(save_dir: str, mel: torch.Tensor):
    from visualize.plot import plot_spectrogram

    os.makedirs(save_dir, exist_ok=True)

    data = mel.squeeze().cpu().numpy()
    fig = plot_spectrogram(data)
    fig.write_image(f"{save_dir}/spectrogram.png")
