from logging import getLogger

from tool.config import Config

logger = getLogger(__name__)


def train_unet(c: Config):
    import lightning as L
    from hydra.utils import instantiate
    from lightning.pytorch.loggers import NeptuneLogger

    from music_controlnet.module.unet import UNet
    from tool.datamodule import FMAMelSpectrogramDataModule

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
    callbacks = (
        [instantiate(callback) for callback in ct.callbacks] if ct.callbacks else None
    )

    criterion = instantiate(ct.criterion) if ct.criterion is not None else None

    logger.info(f"Trainer logger: {trainer_logger}")
    logger.info(f"Callbacks: {[type(callback) for callback in (callbacks or [])]}")

    model = UNet(lr=ct.lr, criterion=criterion)

    trainer = L.Trainer(
        max_epochs=ct.epochs,
        fast_dev_run=ct.fast_dev_run,
        logger=trainer_logger,
        callbacks=callbacks,
        profiler=ct.profiler,
    )
    trainer.fit(model, datamodule=datamodule)

    if isinstance(trainer_logger, NeptuneLogger):
        trainer_logger.log_model_summary(model=model, max_depth=-1)
        trainer_logger.log_hyperparams(dict(c))  # type: ignore
