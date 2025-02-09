import os
from collections.abc import Callable

import torch
import torchaudio
from torch.utils.data import Dataset

from guitar_set.annotation import GuitarSetAnnotation
from tool.dataset import Resampler

_Return = dict[str, torch.Tensor | str]
_Transform = Callable[[torch.Tensor], _Return]


class GuitarSetDataset(Dataset[_Return]):
    def __init__(
        self,
        *,
        annotation_dir: str,
        audio_dir: str,
        transform: _Transform | None = None,
        sample_rate: int,
    ):
        super().__init__()

        self.annotation_dir = annotation_dir
        self.audio_dir = audio_dir
        self.transform = transform

        self.annotation = GuitarSetAnnotation(annotation_dir)
        self.resampler = Resampler(sample_rate)

        self.audio_paths = sorted(
            [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if "comp" in f]
        )

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, index):
        audio_path = self.audio_paths[index]

        waveform, sr = self._load(audio_path)
        waveform = self.resampler(waveform, sr)
        transformed = self._transform(waveform)

        return {
            **transformed,
            "audio_path": audio_path,
        }

    def _transform(self, waveform):
        return self.transform(waveform) if self.transform else waveform

    def _load(self, audio_path: str) -> tuple[torch.Tensor, int]:
        waveform, sr = torchaudio.load(audio_path)
        return waveform, sr
