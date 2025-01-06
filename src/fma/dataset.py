from typing import Any

import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from fma.metadata import (
    PADDING_INDEX,
    FMAMetadata,
    fma_audio_path,
)
from fma.types import FMADatasetReturn, Transform


def collate_fn(batch: list[FMADatasetReturn]) -> dict[str, Any]:
    keys = batch[0].keys()

    batched_genres = pad_sequence(
        [item["genres"] for item in batch],  # type: ignore
        batch_first=True,
        padding_value=PADDING_INDEX,
    )

    other_keys = {}

    for key in keys:
        if key == "genres":
            continue
        elif isinstance(batch[0][key], torch.Tensor):
            other_keys[key] = torch.stack([item[key] for item in batch])  # type: ignore
        elif isinstance(batch[0][key], str):
            other_keys[key] = [item[key] for item in batch]
        else:
            raise ValueError(
                f"Unsupported data type for key '{key}': {type(batch[0][key])}"
            )

    return {**other_keys, "genres": batched_genres}


class FMADataset(Dataset[FMADatasetReturn]):
    def __init__(
        self,
        *,
        metadata_dir: str,
        audio_dir: str,
        sample_rate: int,
        n_segments: int = 1,
        transform: Transform | None = None,
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
        # return 3 * self.splitter.n_segments

    def __getitem__(self, index) -> FMADatasetReturn:
        id_index = self.splitter.get_id_index(index)

        track_id: int = self.metadata.ids[id_index]
        audio_path = fma_audio_path(self.audio_dir, track_id)

        waveform, sr = self._load(audio_path)
        waveform_splitted = self.splitter(waveform, index)
        waveform_resampled = self.resampler(waveform_splitted, sr)
        transformed = self._transform(waveform_resampled)

        genres = self.metadata.to_genre_indics(id_index)

        return {
            **transformed,
            "genres": genres,
            "audio_path": audio_path,
        }

    def _transform(self, waveform):
        return self.transform(waveform) if self.transform else waveform

    def _load(self, audio_path: str) -> tuple[torch.Tensor, int]:
        waveform, sr = torchaudio.load(audio_path)
        return waveform, sr


class Resampler:
    def __init__(self, sample_rate: int):
        self.target_sample_rate = sample_rate

        # for instance cache
        self.resamples: dict[int, torchaudio.transforms.Resample] = {}

    def __call__(self, waveform: torch.Tensor, source_sample_rate: int) -> torch.Tensor:
        if source_sample_rate != self.target_sample_rate:
            resample = self.get_resample(source_sample_rate)
            waveform = resample(waveform)
        return waveform

    def get_resample(self, source_sample_rate: int):
        resample = self.resamples.setdefault(
            source_sample_rate,
            torchaudio.transforms.Resample(
                source_sample_rate,
                self.target_sample_rate,
            ),
        )

        return resample


class Splitter:
    def __init__(self, n_segments: int):
        self.n_segments = n_segments

    def get_id_index(self, id: int) -> int:
        return id // self.n_segments

    def __call__(self, waveform: torch.Tensor, index: int) -> torch.Tensor:
        segment_index = index % self.n_segments

        length = waveform.size(-1) // self.n_segments

        start = segment_index * length
        end = start + length

        return waveform[..., start:end]
