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


def _instantiate_list(config: list[dict] | None) -> list[Any] | None:
    if config is None:
        return None
    return [instantiate(c) for c in config]


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

    enable_default_callbacks: bool = True

    trainer_logger: dict | None = None
    callbacks: list[dict] | None = None

    # debug
    fast_dev_run: bool = False

    @cached_property
    def criterion_object(self) -> Any | None:
        return instantiate(self.criterion)

    @cached_property
    def trainer_logger_object(self) -> Any | None:
        return instantiate(self.trainer_logger)

    @cached_property
    def callbacks_objects(self) -> list[Any] | None:
        return _instantiate_list(self.callbacks)


@dataclass(frozen=True)
class InferConfig:
    checkpoint_path: str | None = None
    output_dir: str = "./output"

    @cached_property
    def save_dir(self) -> str:
        current_datetime = datetime.now()
        return f"{self.output_dir}/{current_datetime:%Y%m%d_%H%M%S}"


@dataclass(frozen=True)
class MelConfig:
    sr: PositiveInt = 22050
    n_fft: PositiveInt = 2048
    win_length: PositiveInt = 2048
    hop_length: PositiveInt = 256
    n_mels: PositiveInt = 128
    top_db: PositiveInt = 80
    n_segments: PositiveInt = 3
    audio_duration: PositiveInt = 30

    @cached_property
    def n_stft(self) -> int:
        return self.n_fft // 2 + 1

    @cached_property
    def fixed_mel_length(self) -> int:
        """
        モデルの入力は8や16の倍数である必要がある場合がある
        最も近い16の倍数になるように調整する
        """
        from tool.functions import nearest_multiple

        duration = self.audio_duration // self.n_segments
        frame_length = self.sr * duration

        mel_estimated_length = frame_length // self.hop_length
        fixed_length = nearest_multiple(mel_estimated_length, multiple=16)
        return fixed_length

    @cached_property
    def fixed_length(self) -> int:
        return self.fixed_mel_length * self.hop_length


@dataclass(frozen=True)
class Config:
    mode: Mode = "train_unet"

    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    infer: InferConfig = field(default_factory=InferConfig)
    mel: MelConfig = field(default_factory=MelConfig)
