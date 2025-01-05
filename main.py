from collections.abc import Callable

import hydra
from omegaconf import DictConfig, OmegaConf

from tool.config import Config, Mode

_MODE_REGISTRY: dict[Mode, Callable[[Config], None]] = {}


def register_mode(mode: Mode):
    def decorator(func: Callable[[Config], None]):
        _MODE_REGISTRY[mode] = func
        return func

    return decorator


@register_mode("train_unet")
def train_unet(cfg: Config):
    from tool.train import train_unet

    train_unet(cfg)


@register_mode("train_diffwave")
def train_diffwave(cfg: Config):
    from tool.train import train_diffwave

    train_diffwave(cfg)


@register_mode("infer_unet")
def infer_unet(cfg: Config):
    from tool.inference import inference_unet

    inference_unet(cfg)


@register_mode("infer_diffwave")
def infer_diffwave(cfg: Config):
    from tool.inference import inference_diffwave

    inference_diffwave(cfg)


@register_mode("infer")
def infer(cfg: Config):
    from tool.inference import inference

    inference(cfg)


@register_mode("doctor")
def doctor(cfg: Config):
    from tool.doctor import doctor

    doctor(cfg)


@register_mode("clean")
def clean(_):
    from tool.clean import clean

    clean()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    container = OmegaConf.to_container(cfg, resolve=True)
    c = Config(**container)  # type: ignore

    mode_function = _MODE_REGISTRY.get(c.mode)
    if mode_function:
        mode_function(c)
    else:
        raise NotImplementedError(f'Mode "{c.mode}"\'s function is not implemented.')


if __name__ == "__main__":
    main()
