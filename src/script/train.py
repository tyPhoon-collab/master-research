from logging import getLogger

from src.script.config import Config

logger = getLogger(__name__)


def train_unet(c: Config):
    import lightning as L
    from hydra.utils import instantiate
    from lightning.pytorch.loggers import NeptuneLogger

    from src.module.data.datamodule import FMAMelSpectrogramDataModule
    from src.module.model.unet import UNet

    ct = c.train
    cm = c.mel

    datamodule = FMAMelSpectrogramDataModule(
        metadata_dir=ct.metadata_dir,
        audio_dir=ct.audio_dir,
        batch_size=ct.batch_size,
        mel_config=cm,
    )

    trainer_logger = (
        instantiate(ct.trainer_logger) if ct.trainer_logger is not None else None
    )
    model_logger = instantiate(ct.model_logger) if ct.model_logger is not None else None
    callbacks = (
        [instantiate(callback) for callback in ct.callbacks] if ct.callbacks else None
    )

    logger.info(f"Trainer logger: {trainer_logger}")
    logger.info(f"Model logger: {model_logger}")
    logger.info(f"Callbacks: {callbacks}")

    model = UNet(
        logger=model_logger,
        lr=ct.lr,
    )

    trainer = L.Trainer(
        max_epochs=ct.epochs,
        fast_dev_run=ct.fast_dev_run,
        logger=trainer_logger,
        callbacks=callbacks,
    )
    trainer.fit(model, datamodule=datamodule)

    if isinstance(trainer_logger, NeptuneLogger):
        trainer_logger.log_model_summary(model=model, max_depth=-1)
        trainer_logger.log_hyperparams(dict(c))  # type: ignore
