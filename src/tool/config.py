from dataclasses import field
from datetime import datetime
from functools import cached_property
from typing import Any, Literal

from hydra.utils import instantiate
from pydantic import PositiveFloat, PositiveInt
from pydantic.dataclasses import dataclass

Mode = Literal[
    "train_unet",
    "train_diffwave",
    "infer_unet",
    "infer_diffwave",
    "infer",
    "doctor",
    "clean",
]


@dataclass(frozen=True)
class DataConfig:
    metadata_dir: str = "./data/FMA/fma_metadata"
    audio_dir: str = "./data/FMA/fma_small"


@dataclass(frozen=True)
class TrainConfig:
    batch_size: PositiveInt = 2
    epochs: PositiveInt = 1
    lr: PositiveFloat = 1e-4
    criterion: dict | None = None
    accumulate_grad_batches: PositiveInt = 1

    profiler: str | None = None

    trainer_logger: dict | None = None
    callbacks: list[dict] | None = None

    # debug
    fast_dev_run: bool = False

    @cached_property
    def criterion_object(self) -> Any | None:
        if self.criterion is None:
            return None

        return instantiate(self.criterion)

    @cached_property
    def trainer_logger_object(self) -> Any | None:
        if self.trainer_logger is None:
            return None

        return instantiate(self.trainer_logger)

    @cached_property
    def callbacks_objects(self) -> list[Any] | None:
        if self.callbacks is None:
            return None

        callbacks = [instantiate(callback) for callback in self.callbacks]

        return callbacks


@dataclass(frozen=True)
class InferConfig:
    checkpoint_path: str | None = None
    callbacks: list[dict] | None = None
    output_dir: str = "./output"

    @cached_property
    def save_dir(self) -> str:
        current_datetime = datetime.now()
        return f"{self.output_dir}/{current_datetime:%Y%m%d_%H%M%S}"

    @cached_property
    def callbacks_objects(self) -> list[Any] | None:
        if self.callbacks is None:
            return None

        from music_controlnet.module.inference_callbacks import StoreDirectory

        callbacks = [instantiate(callback) for callback in self.callbacks]
        for callback in callbacks or []:
            if isinstance(callback, StoreDirectory):
                callback.set_dir(self.save_dir)

        return callbacks


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

    @cached_property
    def n_stft(self) -> int:
        return self.n_fft // 2 + 1


@dataclass(frozen=True)
class Config:
    mode: Mode = "train_unet"

    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    infer: InferConfig = field(default_factory=InferConfig)
    mel: MelConfig = field(default_factory=MelConfig)
