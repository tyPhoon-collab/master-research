from logging import getLogger

import torch
from meldataset import mel_spectrogram
from torchaudio.transforms import (
    AmplitudeToDB,
    GriffinLim,
    InverseMelScale,
    MelSpectrogram,
)

from .db import DBToAmplitude
from .functions import fixed_time_axis_length
from .util import Clamp, Lambda, Mono, Scale, TrimOrPad

logger = getLogger(__name__)


class BigVGANMel(torch.nn.Module):
    def __init__(
        self,
        audio_duration: int,
        n_segments: int,
        n_fft: int,
        num_mels: int,
        sampling_rate: int,
        hop_size: int,
        win_size: int,
        fmin: int,
        fmax: int,
    ):
        super().__init__()

        fixed_length = fixed_time_axis_length(
            audio_duration=audio_duration,
            n_segments=n_segments,
            sample_rate=sampling_rate,
            hop_length=hop_size,
        )

        self.transform = torch.nn.Sequential(
            Mono(),
            Lambda(
                lambda x: mel_spectrogram(
                    x,
                    n_fft=n_fft,
                    num_mels=num_mels,
                    sampling_rate=sampling_rate,
                    hop_size=hop_size,
                    win_size=win_size,
                    fmin=fmin,
                    fmax=fmax,
                )
            ),
            TrimOrPad(target_length=fixed_length, mode="replicate"),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.transform(waveform)


class Mel(torch.nn.Module):
    def __init__(
        self,
        audio_duration: int,
        n_segments: int = 1,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        win_length: int = 1024,
        hop_length: int = 256,
        n_mels: int = 128,
        top_db: int = 80,
        **kwargs,  # for pass another args
    ):
        super().__init__()

        if kwargs:
            logger.warning(f"Unused kwargs provided: {kwargs}")

        fixed_length = fixed_time_axis_length(
            audio_duration=audio_duration,
            n_segments=n_segments,
            sample_rate=sample_rate,
            hop_length=hop_length,
        )

        self.transform = torch.nn.Sequential(
            Mono(),
            Clamp.one(),
            MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                n_mels=n_mels,
                power=2.0,
                normalized=True,
            ),
            TrimOrPad(target_length=fixed_length, mode="replicate"),
            AmplitudeToDB(
                stype="power",
                top_db=top_db,
            ),
            Scale.one(),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.transform(waveform)


class InverseMel(torch.nn.Module):
    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        win_length: int = 1024,
        hop_length: int = 256,
        n_mels: int = 128,
        top_db: int = 80,
    ):
        super().__init__()

        self.transform = torch.nn.Sequential(
            Lambda(lambda mel: mel / 2 * top_db),
            DBToAmplitude(),
            InverseMelScale(
                n_stft=n_fft // 2 + 1,
                n_mels=n_mels,
                sample_rate=sample_rate,
            ),
            GriffinLim(
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                power=2.0,
            ),
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.transform(mel)
