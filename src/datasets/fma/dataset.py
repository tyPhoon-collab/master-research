from collections.abc import Callable

import torch
import torchaudio
from torch.utils.data import Dataset

from datasets.resampler import Resampler
from datasets.splitter import Splitter

from .metadata import (
    FMAMetadata,
    fma_audio_path,
)

_Return = dict[str, torch.Tensor | str]
_Transform = Callable[[torch.Tensor], _Return]


class FMADataset(Dataset[_Return]):
    def __init__(
        self,
        *,
        metadata_dir: str,
        audio_dir: str,
        sample_rate: int,
        n_segments: int = 1,
        transform: _Transform | None = None,
    ):
        super().__init__()

        self.audio_dir = audio_dir
        self.transform = transform

        self.metadata = FMAMetadata(
            metadata_dir=metadata_dir,
            audio_dir=audio_dir,
        )
        self.splitter = Splitter(n_segments)
        self.resampler = Resampler(sample_rate)

    def __len__(self):
        return len(self.metadata) * self.splitter.n_segments

    def __getitem__(self, index) -> _Return:
        id_index = self.splitter.get_id_index(index)

        track_id: int = self.metadata.ids[id_index]
        audio_path = fma_audio_path(self.audio_dir, track_id)

        waveform, sr = self._load(audio_path)
        waveform = self.splitter(waveform, index)
        waveform = self.resampler(waveform, sr)
        transformed = self._transform(waveform)

        genres = self.metadata.to_genre_indics(id_index)

        return {
            **transformed,
            "genre": genres[0],
            "audio_path": audio_path,
        }

    def _transform(self, waveform):
        return self.transform(waveform) if self.transform else waveform

    def _load(self, audio_path: str) -> tuple[torch.Tensor, int]:
        waveform, sr = torchaudio.load(audio_path)
        return waveform, sr
