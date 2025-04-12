from logging import getLogger

import torch

from .functions import fixed_mel_length, fixed_waveform_length
from .util import Clamp, Mono, TrimOrPad

logger = getLogger(__name__)


class NormalizeWaveform(torch.nn.Module):
    def __init__(
        self,
        audio_duration: int,
        n_segments: int = 1,
        sample_rate: int = 22050,
        hop_length: int = 256,
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
            Mono(),
            Clamp.one(),
            TrimOrPad(target_length=fixed_length),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.transform(waveform)


class DynamicNormalize(torch.nn.Module):
    def __init__(self, clip_val: float = 1e-5, C: float = 1.0):
        super().__init__()
        self.clip_val = clip_val
        self.C = C

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ref: bigvgan.meldataset.dynamic_range_compression_torch
        return torch.log(torch.clamp(x, min=self.clip_val) * self.C)


class DynamicDenormalize(torch.nn.Module):
    def __init__(self, C: float = 1.0):
        super().__init__()
        self.C = C

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ref: bigvgan.meldataset.dynamic_range_compression_torch
        return torch.exp(x) / self.C
