import json
import os
from collections.abc import Callable
from os import PathLike
from typing import TypeVar

import polars as pl
import torch
import torchaudio
from torch.utils.data import Dataset

from loader.audio_loader import AudioLoader, TorchAudioLoader

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
        audio_loader: AudioLoader | None = None,  # default is torchaudio loader
    ):
        super().__init__()

        metadata_path = os.path.join(metadata_dir, "raw_tracks.csv")
        all_metadata = pl.read_csv(metadata_path)

        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.transform = transform

        self.audio_loader: AudioLoader = audio_loader or TorchAudioLoader()

        # check audio file exists and filter
        exists_metadata = all_metadata.with_columns(
            all_metadata["track_id"]
            .map_elements(
                lambda id: os.path.exists(self._to_audio_path(id)),
                return_dtype=pl.Boolean,
            )
            .alias("exists")
        ).filter(pl.col("exists"))

        if len(exists_metadata) == 0:
            raise FileNotFoundError("No valid audio file found")

        # [mdeff/fma Wiki](https://github.com/mdeff/fma/wiki#excerpts-shorter-than-30s-and-erroneous-audio-length-metadata)
        # 不完全なデータあるため、それらは削除する
        # fma_small/098/098565.mp3 => 1.6s
        # fma_small/098/098567.mp3 => 0.5s
        # fma_small/098/098569.mp3 => 1.5s
        # fma_small/099/099134.mp3 => 0s
        # fma_small/108/108925.mp3 => 0s
        # fma_small/133/133297.mp3 => 0s

        invalid_ids = [98565, 98567, 98569, 99134, 108925, 133297]
        self.ids = exists_metadata.filter(~pl.col("track_id").is_in(invalid_ids))[
            "track_id"
        ]

        self.genres = exists_metadata["track_genres"]

        df_genres = pl.read_csv(os.path.join(metadata_dir, "genres.csv"))

        # track_genresに含まれるgenre_titleに表記揺れがありそうなので、idで管理する
        self.genre_index_map = {
            str(id): i for i, id in enumerate(df_genres["genre_id"])
        }

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
        audio_path = self._to_audio_path(id)

        waveform, sr = self.audio_loader.load(audio_path)
        resampled_waveform = self._resample(sr, waveform)
        transformed = self._transform(resampled_waveform)

        genres = self._to_genre_indics(index)

        return transformed, genres

    def _to_genre_indics(self, index) -> Genres:
        raw_genre: str = self.genres[index]
        genres = json.loads(
            raw_genre.replace("'", '"'),
        )

        genre_ids = [genre["genre_id"] for genre in genres]

        return torch.tensor([self.genre_index_map[id] for id in genre_ids])

    def _transform(self, waveform):
        return self.transform(waveform) if self.transform else waveform

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

    def _to_audio_path(self, id: int) -> str:
        track_id = f"{id:06d}"  # 0 filled 6 digits str
        folder = track_id[:3]
        audio_path = os.path.join(self.audio_dir, folder, f"{track_id}.mp3")

        return audio_path
