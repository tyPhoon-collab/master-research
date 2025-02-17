from logging import getLogger
from typing import Literal

import torch
import torch.nn.functional as F
import torchaudio
from torchaudio.transforms import (
    AmplitudeToDB,
    GriffinLim,
    InverseMelScale,
    MelSpectrogram,
)

from tool.functions import fixed_mel_length, fixed_waveform_length

logger = getLogger(__name__)


class MelTransform:
    def __init__(self, **kwargs):
        self.mel = Mel(**kwargs)

    def __call__(self, x: torch.Tensor) -> dict:
        return {
            "mel": self.mel(x),
        }


class MelWaveformTransform:
    def __init__(self, **kwargs):
        self.mel = Mel(**kwargs)
        self.waveform = NormalizeWaveform(**kwargs)

    def __call__(self, x: torch.Tensor) -> dict:
        return {
            "mel": self.mel(x),
            "waveform": self.waveform(x),
        }


class NormalizeWaveform(torch.nn.Module):
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


class Mel(torch.nn.Module):
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


class InverseMel(torch.nn.Module):
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


class TrimOrPad(torch.nn.Module):
    def __init__(
        self,
        target_length: int,
        mode: Literal["constant", "reflect", "replicate"] = "constant",
    ):
        super().__init__()
        self.target_length = target_length
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_length = self.target_length
        current_length = x.size(-1)

        if current_length > target_length:
            x = x[..., :target_length]
        elif current_length < target_length:
            x = F.pad(x, (0, target_length - current_length), mode=self.mode)

        return x


class ToMono(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform


class Lambda(torch.nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


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


class Clamp(torch.nn.Module):
    def __init__(self, min: float, max: float):
        super().__init__()
        self.min = min
        self.max = max

    @classmethod
    def one(cls) -> "Clamp":
        return cls(-1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, min=self.min, max=self.max)


class Scale(torch.nn.Module):
    def __init__(
        self,
        to: tuple[float, float],
        min: float | None = None,
        max: float | None = None,
    ):
        super().__init__()

        self.min = min
        self.max = max
        self.to_ = to

    @classmethod
    def one(cls) -> "Scale":
        return cls(to=(-1, 1))

    @classmethod
    def db_normalize(cls, top_db: float) -> "Scale":
        return cls(min=-top_db, max=0, to=(-1, 1))

    @classmethod
    def db_denormalize(cls, top_db: float) -> "Scale":
        return cls(min=-1, max=1, to=(-top_db, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t_min, t_max = self.to_
        max = self.max or x.max()
        min = self.min or x.min()

        range = max - min

        if range == 0:
            return x

        return (x - min) / range * (t_max - t_min) + t_min


class DBToAmplitude(torch.nn.Module):
    def __init__(self, ttype: Literal["power", "magnitude"] = "power") -> None:
        super().__init__()
        self.ttype = ttype

    def forward(self, db: torch.Tensor) -> torch.Tensor:
        return torchaudio.functional.DB_to_amplitude(
            db,
            ref=1.0,
            power=1 if self.ttype == "power" else 0.5,
        )
