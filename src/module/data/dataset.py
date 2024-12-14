import json
import os
from collections.abc import Callable
from os import PathLike
from typing import TypeVar

import polars as pl
import torch
import torchaudio
from torch.utils.data import Dataset

Genres = torch.Tensor
FMADatasetReturn = tuple[torch.Tensor, Genres]
InputType = TypeVar("InputType", bound=torch.Tensor)
OutputType = TypeVar("OutputType")
Transform = Callable[[InputType], OutputType]


class FMADataset(Dataset[FMADatasetReturn]):
    PADDING_INDEX = 163
    NUM_GENRES = 163 + 1

    def __init__(
        self,
        *,
        metadata_dir: PathLike | str,
        audio_dir: PathLike | str,
        sample_rate: int,
        transform: Transform | None = None,
        audio_duration: int = 30,
    ):
        super().__init__()

        metadata_path = os.path.join(metadata_dir, "raw_tracks.csv")
        self.all_metadata = pl.read_csv(metadata_path)
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.transform = transform
        self.audio_duration = audio_duration

        # check audio file exists and filter
        self.metadata = self.all_metadata.with_columns(
            self.all_metadata["track_id"]
            .map_elements(self._exists, return_dtype=pl.Boolean)
            .alias("exists")
        ).filter(pl.col("exists"))

        if len(self.metadata) == 0:
            raise FileNotFoundError("No valid audio file found")

        self.genres = self.metadata["track_genres"]
        self.ids = self.metadata["track_id"]

        df_genres = pl.read_csv(os.path.join(metadata_dir, "genres.csv"))
        self.genre_index_map = {genre: i for i, genre in enumerate(df_genres["title"])}

        # for instance cache
        self.resamples: dict[int, torchaudio.transforms.Resample] = {}

    @classmethod
    def collate_fn(cls, batch: list[FMADatasetReturn]) -> FMADatasetReturn:
        waveforms, genres = zip(*batch)

        batch_waveforms = torch.stack(waveforms)

        max_genre_dim = max(genre.size(0) for genre in genres)

        batch_genres = torch.full(
            (len(genres), max_genre_dim),
            cls.PADDING_INDEX,
        )

        for i, genre in enumerate(genres):
            batch_genres[i, : genre.size(0)] = genre

        return batch_waveforms, batch_genres

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index) -> FMADatasetReturn:
        id: int = self.ids[index]

        waveform, sr = self._load(id)
        waveform = self._resample(sr, waveform)
        waveform = self._mono(waveform)
        waveform = self._trim_or_pad_waveform(waveform)

        transformed = self._transform(waveform)
        genres = self._convert_to_genre_indics(index)

        return transformed, genres

    def _load(self, id):
        audio_path = self._build_audio_path(id)

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            waveform, sr = torchaudio.load(audio_path)
        except RuntimeError:
            raise RuntimeError(f"Failed to load audio file: {audio_path}")
        return waveform, sr

    def _convert_to_genre_indics(self, index) -> Genres:
        raw_genre: str = self.genres[index]
        genres = json.loads(
            raw_genre.replace("'", '"'),
        )

        titles = [genre["genre_title"] for genre in genres]

        return torch.tensor([self.genre_index_map[title] for title in titles])

    def _transform(self, waveform):
        return self.transform(waveform) if self.transform else waveform

    def _mono(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform

    def _resample(self, sr: int, waveform: torch.Tensor):
        if sr != self.sample_rate:
            resample = self.resamples.setdefault(
                sr,
                torchaudio.transforms.Resample(
                    sr,
                    self.sample_rate,
                ),
            )
            waveform = resample(waveform)
        return waveform

    def _build_audio_path(self, id: int) -> str:
        track_id = f"{id:06d}"  # 0 filled 6 digits str
        folder = track_id[:3]
        audio_path = os.path.join(self.audio_dir, folder, f"{track_id}.mp3")

        return audio_path

    def _exists(self, id: int) -> bool:
        audio_path = self._build_audio_path(id)
        return os.path.exists(audio_path)

    def _trim_or_pad_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        target_length = self.sample_rate * self.audio_duration
        current_length = waveform.size(-1)

        if current_length > target_length:
            waveform = waveform[..., :target_length]
        elif current_length < target_length:
            padding = target_length - current_length
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        return waveform
