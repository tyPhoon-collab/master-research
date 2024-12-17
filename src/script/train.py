from dataclasses import asdict

from hydra.utils import instantiate
from pytorch_lightning.loggers import NeptuneLogger

from src.script.config import TrainConfig


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

    trainer_logger = (
        instantiate(cfg.trainer_logger) if cfg.trainer_logger is not None else None
    )
    model_logger = (
        instantiate(cfg.model_logger) if cfg.model_logger is not None else None
    )
    callbacks = [instantiate(callback) for callback in cfg.callbacks]

    print("trainer_logger:", trainer_logger)
    print("model_logger:", model_logger)
    print("callbacks:", callbacks)

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

    if trainer_logger is NeptuneLogger:
        trainer_logger.log_model_summary(model=model, max_depth=-1)
        trainer_logger.log_hyperparams(asdict(cfg))
