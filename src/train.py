from dataclasses import asdict

from hydra.utils import instantiate
from pytorch_lightning.loggers import NeptuneLogger

from src.config import TrainConfig


def train_unet(cfg: TrainConfig):
    import lightning as L

    from src.module.data.datamodule import FMAMelSpectrogramDataModule
    from src.module.model.unet import UNet

    datamodule = FMAMelSpectrogramDataModule(
        metadata_dir=cfg.metadata_dir,
        audio_dir=cfg.audio_dir,
        batch_size=cfg.batch_size,
        sample_rate=cfg.sr,
    )
    model = UNet()

    logger = instantiate(cfg.logger) if cfg.logger is not None else None

    trainer = L.Trainer(
        max_epochs=cfg.epochs,
        fast_dev_run=cfg.fast_dev_run,
        logger=logger,
    )
    trainer.fit(model, datamodule=datamodule)

    if logger is NeptuneLogger:
        logger.log_model_summary(model=model, max_depth=-1)
        logger.log_hyperparams(asdict(cfg))
