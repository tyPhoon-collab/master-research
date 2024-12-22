from typing import Literal

import torch
import torchaudio.functional as F


class TrimOrPad(torch.nn.Module):
    def __init__(self, target_length: int):
        super().__init__()
        self.target_length = target_length

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        target_length = self.target_length
        current_length = waveform.size(-1)

        if current_length > target_length:
            waveform = waveform[..., :target_length]
        elif current_length < target_length:
            padding = target_length - current_length
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        return waveform


class ToMono(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform


class NormalizeMinusOneToOne(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        min = waveform.min()
        max = waveform.max()
        return (waveform - min) / (max - min) * 2 - 1


class Lambda(torch.nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class DBToAmplitude(torch.nn.Module):
    def __init__(self, ttype: Literal["power", "magnitude"] = "power") -> None:
        super().__init__()
        self.ttype = ttype

    def forward(self, db: torch.Tensor) -> torch.Tensor:
        return F.DB_to_amplitude(
            db,
            ref=1.0,
            power=1 if self.ttype == "power" else 0.5,
        )
