from collections.abc import Callable
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from tool.config import Config, Mode


class _Handler:
    @staticmethod
    def train_unet(cfg: Config):
        from tool.train import train_unet

        train_unet(cfg)

    @staticmethod
    def train_diffwave(cfg: Config):
        from tool.train import train_diffwave

        train_diffwave(cfg)

    @staticmethod
    def infer_unet(cfg: Config):
        from tool.inference import inference_unet

        inference_unet(cfg)

    @staticmethod
    def infer(cfg: Config):
        from tool.inference import inference

        inference(cfg)

    @staticmethod
    def doctor(cfg: Config):
        from tool.doctor import doctor

        doctor(cfg)

    @staticmethod
    def clean(_):
        from tool.clean import clean

        clean()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    container: Any = OmegaConf.to_container(cfg, resolve=True)
    c = Config(**container)

    mode_methods: dict[Mode, Callable[[Config], None]] = {
        "train_unet": _Handler.train_unet,
        "train_diffwave": _Handler.train_diffwave,
        "infer_unet": _Handler.infer_unet,
        "infer": _Handler.infer,
        "doctor": _Handler.doctor,
        "clean": _Handler.clean,
    }

    mode_methods[c.mode](c)


if __name__ == "__main__":
    main()
