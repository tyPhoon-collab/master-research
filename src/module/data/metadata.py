import ast
import os

import polars as pl
import torch

from src.loader.csv_loader import load_multi_header_csv
from src.types_ import Genres

PADDING_INDEX = 163
NUM_GENRES = 163 + 1


def fma_audio_path(audio_dir: str, track_id: int) -> str:
    id = f"{track_id:06d}"  # 0 filled 6 digits str
    folder_name = id[:3]
    audio_path = os.path.join(audio_dir, folder_name, f"{track_id}.mp3")

    return audio_path


# [mdeff/fma Wiki](https://github.com/mdeff/fma/wiki#excerpts-shorter-than-30s-and-erroneous-audio-length-metadata)
# 不完全なデータあるため、それらは削除する
# fma_small/098/098565.mp3 => 1.6s
# fma_small/098/098567.mp3 => 0.5s
# fma_small/098/098569.mp3 => 1.5s
# fma_small/099/099134.mp3 => 0s
# fma_small/108/108925.mp3 => 0s
# fma_small/133/133297.mp3 => 0s
_default_ignore_ids = [98565, 98567, 98569, 99134, 108925, 133297]


class FMAMetadata:
    def __init__(
        self,
        metadata_dir: str,
        audio_dir: str,
        ignore_ids: list[int] | None = _default_ignore_ids,
    ):
        metadata_path = os.path.join(metadata_dir, "tracks.csv")
        metadata_all = load_multi_header_csv(metadata_path, 3)

        metadata_exists = metadata_all.with_columns(
            metadata_all["track_id"]
            .map_elements(
                lambda id: os.path.exists(fma_audio_path(audio_dir, id)),
                return_dtype=pl.Boolean,
            )
            .alias("exists")
        ).filter(pl.col("exists"))

        if len(metadata_exists) == 0:
            raise FileNotFoundError("No valid audio file found")

        self.ids = metadata_exists.filter(~pl.col("track_id").is_in(ignore_ids or []))[
            "track_id"
        ]
        self.str_genre_ids = metadata_exists["track_genres"]

        genre_ids = pl.read_csv(os.path.join(metadata_dir, "genres.csv"))["genre_id"]
        self.genre_id_to_index_map = {id: i for i, id in enumerate(genre_ids)}

    def to_genre_indics(self, index) -> Genres:
        genre_ids: list[int] = ast.literal_eval(self.str_genre_ids[index])

        return torch.tensor([self.genre_id_to_index_map[id] for id in genre_ids])

    def __len__(self):
        return len(self.ids)
