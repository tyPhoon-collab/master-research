from logging import getLogger

from src.script.config import TrainConfig

logger = getLogger(__name__)


def train_unet(cfg: TrainConfig):
    from dataclasses import asdict

    import lightning as L
    from hydra.utils import instantiate
    from lightning.pytorch.loggers import NeptuneLogger

    from src.module.data.datamodule import FMAMelSpectrogramDataModule
    from src.module.model.unet import UNet

    datamodule = FMAMelSpectrogramDataModule(
        metadata_dir=cfg.metadata_dir,
        audio_dir=cfg.audio_dir,
        batch_size=cfg.batch_size,
        sample_rate=cfg.sr,
    )

    trainer_logger = (
        instantiate(cfg.trainer_logger) if cfg.trainer_logger is not None else None
    )
    model_logger = (
        instantiate(cfg.model_logger) if cfg.model_logger is not None else None
    )
    callbacks = (
        [instantiate(callback) for callback in cfg.callbacks] if cfg.callbacks else None
    )

    logger.info(f"Trainer logger: {trainer_logger}")
    logger.info(f"Model logger: {model_logger}")
    logger.info(f"Callbacks: {callbacks}")

    model = UNet(
        logger=model_logger,
        lr=cfg.lr,
    )

    trainer = L.Trainer(
        max_epochs=cfg.epochs,
        fast_dev_run=cfg.fast_dev_run,
        logger=trainer_logger,
        callbacks=callbacks,
    )
    trainer.fit(model, datamodule=datamodule)

    if isinstance(trainer_logger, NeptuneLogger):
        trainer_logger.log_model_summary(model=model, max_depth=-1)
        trainer_logger.log_hyperparams(asdict(cfg))
