import hydra
from hydra.core.config_store import ConfigStore

from tool.config import Config, Mode

cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)


class _Handler:
    @staticmethod
    def train_unet(cfg):
        from tool.train import train_unet

        train_unet(cfg)

    @staticmethod
    def infer_unet(cfg):
        from tool.inference import inference_unet

        inference_unet(cfg)

    @staticmethod
    def infer(cfg):
        from tool.inference import inference

        inference(cfg)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: Config):
    mode_methods = {
        Mode.train_unet.name: _Handler.train_unet,
        Mode.infer_unet.name: _Handler.infer_unet,
        Mode.infer.name: _Handler.infer,
    }

    if cfg.mode not in mode_methods:
        raise ValueError(f"Invalid mode: {cfg.mode}")

    mode_methods[cfg.mode](cfg)


if __name__ == "__main__":
    main()
