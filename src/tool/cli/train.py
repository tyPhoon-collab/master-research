from logging import getLogger

from callback.optimizer_hooks import ScheduleFreeOptimizerCallback
from tool.config import Config

logger = getLogger(__name__)


def train_unet(c: Config):
    from music_controlnet.module.unet import UNetLightning
    from tool.factory import build_unet_datamodule

    ct = c.train

    datamodule = build_unet_datamodule(c)

    model = UNetLightning(lr=ct.lr, criterion=ct.criterion_object)

    _train(c, datamodule, model)


def train_diffwave(c: Config):
    from tool.factory import build_vocoder_datamodule
    from vocoder.module.diffwave import DiffWaveLightning

    ct = c.train

    datamodule = build_vocoder_datamodule(c)

    model = DiffWaveLightning(
        n_mels=c.mel.n_mels,
        lr=ct.lr,
        criterion=ct.criterion_object,
    )

    _train(c, datamodule, model)


def train_music_hifi(c: Config):
    from tool.factory import build_vocoder_datamodule
    from vocoder.module.music_hifi import MusicHiFiLightning

    ct = c.train

    datamodule = build_vocoder_datamodule(c)

    model = MusicHiFiLightning(
        lr=ct.lr,
        sampling_rate=c.mel.sr,
        n_mels=c.mel.n_mels,
    )

    _train(c, datamodule, model)


def _train(c: Config, datamodule, model):
    import lightning as L
    from lightning.pytorch.loggers import NeptuneLogger

    ct = c.train

    trainer_logger = ct.trainer_logger_object
    callbacks = ct.callbacks_objects or []

    if ct.enable_default_callbacks:
        callbacks.insert(0, ScheduleFreeOptimizerCallback())

    logger.info(f"Trainer logger: {trainer_logger.__class__.__name__}")
    logger.info(f"Callbacks: {[callback.__class__.__name__ for callback in callbacks]}")

    trainer = L.Trainer(
        max_epochs=ct.epochs,
        fast_dev_run=ct.fast_dev_run,
        logger=trainer_logger,
        callbacks=callbacks,
        profiler=ct.profiler,
        accumulate_grad_batches=ct.accumulate_grad_batches,
    )
    trainer.fit(model, datamodule=datamodule)

    if isinstance(trainer_logger, NeptuneLogger):
        trainer_logger.log_model_summary(model=model, max_depth=-1)
        trainer_logger.log_hyperparams(c.model_dump())
