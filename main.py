import hydra
from hydra.core.config_store import ConfigStore

from music_controlnet.script.config import Config, Mode

cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: Config):
    match cfg.mode:
        case Mode.train_unet.name:
            from music_controlnet.script.train import train_unet

            train_unet(cfg)
        case Mode.infer_unet.name:
            from music_controlnet.script.inference import inference_unet

            inference_unet(cfg)

        case Mode.infer.name:
            from music_controlnet.script.inference import inference

            inference(cfg)
        case _:
            raise ValueError(f"Invalid mode: {cfg.mode}")


if __name__ == "__main__":
    main()
