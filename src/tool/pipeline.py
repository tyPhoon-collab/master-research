from logging import getLogger
from typing import Any

import torch
from torchaudio.transforms import (
    AmplitudeToDB,
    GriffinLim,
    InverseMelScale,
    MelSpectrogram,
)

from tool.functions import fixed_mel_length, fixed_waveform_length
from tool.transforms import (
    Clamp,
    DBToAmplitude,
    Lambda,
    Scale,
    ToMono,
    TrimOrPad,
)

logger = getLogger(__name__)


class MelTransform:
    def __init__(self, **kwargs):
        self.mel = MelSpectrogramPipeline(**kwargs)

    def __call__(self, x: torch.Tensor) -> Any:
        return {
            "mel": self.mel(x),
        }


class MelWaveformTransform:
    def __init__(self, **kwargs):
        self.mel = MelSpectrogramPipeline(**kwargs)
        self.waveform = WaveformPipeline(**kwargs)

    def __call__(self, x: torch.Tensor) -> Any:
        return {
            "mel": self.mel(x),
            "waveform": self.waveform(x),
        }


class WaveformPipeline(torch.nn.Module):
    def __init__(
        self,
        audio_duration: int,
        n_segments: int,
        sample_rate: int,
        hop_length: int,
        **kwargs,  # for pass another args
    ):
        super().__init__()

        if kwargs:
            logger.warning(f"Unused kwargs provided: {kwargs}")

        fixed_length = fixed_waveform_length(
            fixed_mel_length=fixed_mel_length(
                audio_duration=audio_duration,
                n_segments=n_segments,
                sample_rate=sample_rate,
                hop_length=hop_length,
            ),
            hop_length=hop_length,
        )

        self.transform = torch.nn.Sequential(
            ToMono(),
            Clamp.one(),
            TrimOrPad(target_length=fixed_length),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.transform(waveform)


class MelSpectrogramPipeline(torch.nn.Module):
    def __init__(
        self,
        audio_duration: int,
        n_segments: int,
        sample_rate: int,
        n_fft: int,
        win_length: int,
        hop_length: int,
        n_mels: int,
        top_db: int,
        **kwargs,  # for pass another args
    ):
        super().__init__()

        if kwargs:
            logger.warning(f"Unused kwargs provided: {kwargs}")

        fixed_length = fixed_mel_length(
            audio_duration=audio_duration,
            n_segments=n_segments,
            sample_rate=sample_rate,
            hop_length=hop_length,
        )

        self.transform = torch.nn.Sequential(
            ToMono(),
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


class InverseMelSpectrogramPipeline(torch.nn.Module):
    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        win_length: int,
        hop_length: int,
        n_mels: int,
        top_db: int,
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
