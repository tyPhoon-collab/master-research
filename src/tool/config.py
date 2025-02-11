from dataclasses import field
from datetime import datetime
from functools import cached_property
from typing import Any, Literal

from hydra.utils import instantiate
from pydantic import BaseModel, PositiveFloat, PositiveInt

Mode = Literal[
    "train",
    "infer",
    "doctor",
    "clean",
    "preprocess",
]


def _instantiate_list(config: list[dict] | None) -> list[Any] | None:
    if config is None:
        return None
    return [instantiate(c) for c in config]


class TrainConfig(BaseModel):
    epochs: PositiveInt = 1
    lr: PositiveFloat = 1e-4

    accumulate_grad_batches: PositiveInt = 1

    profiler: str | None = None

    enable_default_callbacks: bool = True

    trainer_logger: dict | None = None
    callbacks: list[dict] | None = None

    # debug
    fast_dev_run: bool = False

    @cached_property
    def trainer_logger_object(self) -> Any | None:
        return instantiate(self.trainer_logger)

    @cached_property
    def callbacks_objects(self) -> list[Any] | None:
        return _instantiate_list(self.callbacks)


class InferConfig(BaseModel):
    generator: dict = field(default_factory=dict)
    vocoder: dict = field(default_factory=dict)

    output_dir: str = "./output"

    @cached_property
    def generator_object(self) -> Any:
        return instantiate(self.generator)

    @cached_property
    def vocoder_object(self) -> Any:
        return instantiate(self.vocoder)

    @cached_property
    def save_dir(self) -> str:
        current_datetime = datetime.now()
        return f"{self.output_dir}/{current_datetime:%Y%m%d_%H%M%S}"


class Config(BaseModel):
    train: TrainConfig = field(default_factory=TrainConfig)
    infer: InferConfig = field(default_factory=InferConfig)
    model: dict = field(default_factory=dict)
    data: dict = field(default_factory=dict)
    preprocess: dict = field(default_factory=dict)

    mode: Mode = "train"

    @cached_property
    def model_object(self) -> Any:
        return instantiate(self.model)

    @cached_property
    def data_object(self) -> Any:
        return instantiate(self.data)

    @cached_property
    def preprocess_object(self) -> Any:
        return instantiate(self.preprocess)
