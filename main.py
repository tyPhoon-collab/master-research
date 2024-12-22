import hydra
from hydra.core.config_store import ConfigStore

from src.script.config import Config

cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: Config):
    match cfg.mode:
        case "train_unet":
            from src.script.train import train_unet

            train_unet(cfg)
        case "infer_unet":
            from src.script.inference import inference_unet

            inference_unet(cfg.infer)
        case _:
            raise ValueError(f"Invalid mode: {cfg.mode}")


if __name__ == "__main__":
    main()
