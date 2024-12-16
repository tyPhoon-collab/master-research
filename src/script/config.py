from dataclasses import dataclass, field
from enum import Enum, auto


@dataclass(frozen=True)
class TrainConfig:
    metadata_dir: str = "./data/FMA/fma_metadata"
    audio_dir: str = "./data/FMA/fma_small"

    sr: int = 22050
    batch_size: int = 2
    epochs: int = 1
    lr: float = 1e-4

    fast_dev_run: bool = False

    trainer_logger: dict | None = None
    model_logger: dict | None = None


@dataclass(frozen=True)
class InferConfig:
    checkpoint_path: str | None = None


class Mode(Enum):
    train_unet = auto()
    infer_unet = auto()


@dataclass(frozen=True)
class Config:
    mode: Mode = Mode.train_unet

    train: TrainConfig = field(default_factory=TrainConfig)
    infer: InferConfig = field(default_factory=InferConfig)
