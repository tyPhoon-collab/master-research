from typing import Literal

import torch
import torch.nn.functional as F


class Mono(torch.nn.Module):
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
