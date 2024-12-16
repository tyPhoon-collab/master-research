import hydra

from src.config import Config


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: Config):
    match cfg.mode:
        case "train_unet":
            from src.train import train_unet

            train_unet(cfg.train)
        case _:
            raise ValueError(f"Invalid mode: {cfg.mode}")


if __name__ == "__main__":
    main()
