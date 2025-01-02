from dataclasses import dataclass, field


@dataclass(frozen=True)
class TrainConfig:
    metadata_dir: str = "./data/FMA/fma_metadata"
    audio_dir: str = "./data/FMA/fma_small"

    batch_size: int = 2
    epochs: int = 1
    lr: float = 1e-4
    criterion: dict | None = None

    profiler: str | None = None

    # debug
    fast_dev_run: bool = False

    # instantiate
    trainer_logger: dict | None = None
    callbacks: list[dict] | None = None


@dataclass(frozen=True)
class InferConfig:
    checkpoint_path: str | None = None
    callbacks: list[dict] | None = None


@dataclass(frozen=True)
class MelConfig:
    sr: int = 22050
    n_fft: int = 2048
    win_length: int = 2048
    hop_length: int = 256
    # sr / hop_length * 30 / num_segments -> 16 * n -> 864 (n = 54)
    fixed_length: int = 864
    n_mels: int = 128
    top_db: int = 80
    num_segments: int = 3


@dataclass(frozen=True)
class Config:
    # Literal["train_unet", "train_diffwave", "infer_unet", "infer"]
    mode: str = "train_unet"

    train: TrainConfig = field(default_factory=TrainConfig)
    infer: InferConfig = field(default_factory=InferConfig)
    mel: MelConfig = field(default_factory=MelConfig)
