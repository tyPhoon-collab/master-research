from dataclasses import dataclass, field

from omegaconf import DictConfig


@dataclass(frozen=True)
class TrainConfig:
    metadata_dir: str = "./data/FMA/fma_metadata"
    audio_dir: str = "./data/FMA/fma_small"

    sr: int = 22050
    batch_size: int = 2
    epochs: int = 1

    fast_dev_run: bool = False

    logger: DictConfig | None = None


@dataclass(frozen=True)
class Config:
    mode: str = "train_unet"

    train: TrainConfig = field(default_factory=TrainConfig)
