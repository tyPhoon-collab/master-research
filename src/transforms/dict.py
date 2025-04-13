import torch

from .mel import Mel
from .normalize import NormalizeWaveform
from .reassign import _Reassigned


class ReassignedTransform:
    def __init__(self, **kwargs):
        self.reassigned = _Reassigned(**kwargs)

    def __call__(self, x: torch.Tensor) -> dict:
        return {"reassigned": self.reassigned(x)}


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
