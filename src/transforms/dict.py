import torch

from .mel import Mel
from .normalize import NormalizeWaveform
from .reassign import Reassigned


class ReassignedTransform:
    def __init__(self, **kwargs):
        self.reassigned = Reassigned(**kwargs)

    def __call__(self, x: torch.Tensor) -> dict:
        return {"spectrogram": self.reassigned(x)}


class ReassignedWaveformTransform:
    def __init__(self, **kwargs):
        self.reassigned = Reassigned(**kwargs)
        self.waveform = NormalizeWaveform(**kwargs)

    def __call__(self, x: torch.Tensor) -> dict:
        return {
            "spectrogram": self.reassigned(x),
            "waveform": self.waveform(x),
        }


class MelTransform:
    def __init__(self, **kwargs):
        self.mel = Mel(**kwargs)

    def __call__(self, x: torch.Tensor) -> dict:
        return {
            "spectrogram": self.mel(x),
        }


class MelWaveformTransform:
    def __init__(self, **kwargs):
        self.mel = Mel(**kwargs)
        self.waveform = NormalizeWaveform(**kwargs)

    def __call__(self, x: torch.Tensor) -> dict:
        return {
            "spectrogram": self.mel(x),
            "waveform": self.waveform(x),
        }
