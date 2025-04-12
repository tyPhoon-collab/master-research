from collections.abc import Callable

import hydra
from omegaconf import DictConfig, OmegaConf

from cli.config import Config, Mode

_MODE_REGISTRY: dict[Mode, Callable[[Config], None]] = {}


def register_mode(mode: Mode):
    def decorator(func: Callable[[Config], None]):
        _MODE_REGISTRY[mode] = func
        return func

    return decorator


@register_mode("train")
def train(cfg: Config):
    from cli.train import train

    train(cfg)


@register_mode("infer")
def infer(cfg: Config):
    from cli.infer import inference

    inference(cfg)


@register_mode("doctor")
def doctor(cfg: Config):
    from cli.doctor import doctor

    doctor(cfg)


@register_mode("clean")
def clean(_):
    from cli.clean import clean

    clean()


@register_mode("preprocess")
def preprocess(cfg: Config):
    from cli.preprocess import preprocess

    preprocess(cfg)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    container = OmegaConf.to_container(cfg, resolve=True)
    c = Config(**container)  # type: ignore

    action = _MODE_REGISTRY.get(c.mode)
    if not action:
        raise NotImplementedError(f'Mode "{c.mode}"\'s function is not implemented.')

    action(c)


if __name__ == "__main__":
    main()
