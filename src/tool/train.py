from logging import getLogger

from tool.config import Config

logger = getLogger(__name__)


def train_unet(c: Config):
    from hydra.utils import instantiate

    from music_controlnet.module.unet import UNetLightning
    from tool.factory import build_unet_datamodule

    ct = c.train

    datamodule = build_unet_datamodule(c)

    criterion = instantiate(ct.criterion) if ct.criterion is not None else None
    model = UNetLightning(lr=ct.lr, criterion=criterion)

    _base_train(c, datamodule, model)


def train_diffwave(c: Config):
    from hydra.utils import instantiate

    from tool.factory import build_diffwave_datamodule
    from vocoder.module.diffwave import DiffWaveLightning

    ct = c.train

    datamodule = build_diffwave_datamodule(c)

    criterion = instantiate(ct.criterion) if ct.criterion is not None else None
    model = DiffWaveLightning(n_mels=c.mel.n_mels, lr=ct.lr, criterion=criterion)

    _base_train(c, datamodule, model)


def _base_train(c: Config, datamodule, model):
    import lightning as L
    from hydra.utils import instantiate
    from lightning.pytorch.loggers import NeptuneLogger

    ct = c.train

    trainer_logger = (
        instantiate(ct.trainer_logger) if ct.trainer_logger is not None else None
    )
    callbacks = (
        [instantiate(callback) for callback in ct.callbacks] if ct.callbacks else None
    )

    logger.info(f"Trainer logger: {trainer_logger}")
    logger.info(f"Callbacks: {[type(callback) for callback in (callbacks or [])]}")

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
