from logging import getLogger

from callback.optimizer_hooks import ScheduleFreeOptimizerCallback
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

    _train(c, datamodule, model)


def train_diffwave(c: Config):
    from hydra.utils import instantiate

    from tool.factory import build_diffwave_datamodule
    from vocoder.module.diffwave import DiffWaveLightning

    ct = c.train

    datamodule = build_diffwave_datamodule(c)

    criterion = instantiate(ct.criterion) if ct.criterion is not None else None
    model = DiffWaveLightning(n_mels=c.mel.n_mels, lr=ct.lr, criterion=criterion)

    _train(c, datamodule, model)


def _train(c: Config, datamodule, model):
    import lightning as L
    from lightning.pytorch.loggers import NeptuneLogger

    ct = c.train

    default_callbacks = [
        ScheduleFreeOptimizerCallback(),
    ]

    trainer_logger = ct.trainer_logger_object
    callbacks = ct.callbacks_objects or []

    if ct.enable_default_callbacks:
        callbacks.extend(default_callbacks)

    logger.info(f"Trainer logger: {trainer_logger}")
    logger.info(f"Callbacks: {[type(callback) for callback in callbacks]}")

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
        trainer_logger.log_hyperparams(dict(c))  # type: ignore
