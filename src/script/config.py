from dataclasses import dataclass, field
from enum import Enum, auto
from functools import cached_property


@dataclass(frozen=True)
class TrainConfig:
    metadata_dir: str = "./data/FMA/fma_metadata"
    audio_dir: str = "./data/FMA/fma_small"

    sr: int = 22050
    batch_size: int = 2
    epochs: int = 1
    lr: float = 1e-4

    # debug
    fast_dev_run: bool = False

    # instantiate
    trainer_logger: dict | None = None
    model_logger: dict | None = None
    callbacks: list[dict] | None = field(default_factory=list)


@dataclass(frozen=True)
class InferConfig:
    checkpoint_path: str | None = None


@dataclass(frozen=True)
class MelConfig:
    sample_rate: int = 22050
    n_fft: int = 2048
    win_length: int = 2048
    hop_length: int = 256
    # sample_rate / hop_length * 30 / num_segments -> 16 * n -> 864 (n = 54)
    fixed_length: int = 864
    n_mels: int = 160
    top_db: int = 80
    num_segments: int = 3

    @cached_property
    def n_stft(self):
        return self.n_fft // 2 + 1


class Mode(Enum):
    train_unet = auto()
    infer_unet = auto()


@dataclass(frozen=True)
class Config:
    mode: Mode = Mode.train_unet

    train: TrainConfig = field(default_factory=TrainConfig)
    infer: InferConfig = field(default_factory=InferConfig)
    mel: MelConfig = field(default_factory=MelConfig)
