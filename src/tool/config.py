from dataclasses import field
from pathlib import Path
from typing import Literal

from pydantic import DirectoryPath, FilePath, PositiveFloat, PositiveInt
from pydantic.dataclasses import dataclass

Mode = Literal["train_unet", "train_diffwave", "infer_unet", "infer", "doctor", "clean"]


@dataclass(frozen=True)
class DataConfig:
    metadata_dir: DirectoryPath = field(default=Path("./data/FMA/fma_metadata"))
    audio_dir: DirectoryPath = field(default=Path("./data/FMA/fma_small"))


@dataclass(frozen=True)
class TrainConfig:
    batch_size: PositiveInt = 2
    epochs: PositiveInt = 1
    lr: PositiveFloat = 1e-4
    criterion: dict | None = None
    accumulate_grad_batches: PositiveInt = 1

    profiler: str | None = None

    # debug
    fast_dev_run: bool = False

    # instantiate
    trainer_logger: dict | None = None
    callbacks: list[dict] | None = None


@dataclass(frozen=True)
class InferConfig:
    checkpoint_path: FilePath | None = None
    callbacks: list[dict] | None = None


@dataclass(frozen=True)
class MelConfig:
    sr: PositiveInt = 22050
    n_fft: PositiveInt = 2048
    win_length: PositiveInt = 2048
    hop_length: PositiveInt = 256
    fixed_length: PositiveInt = 864
    n_mels: PositiveInt = 128
    top_db: PositiveInt = 80
    num_segments: PositiveInt = 3


@dataclass(frozen=True)
class Config:
    mode: Mode = "train_unet"

    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    infer: InferConfig = field(default_factory=InferConfig)
    mel: MelConfig = field(default_factory=MelConfig)
