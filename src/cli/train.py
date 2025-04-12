from logging import getLogger

from callback.optimizer_hooks import ScheduleFreeOptimizerCallback
from cli.config import Config

logger = getLogger(__name__)


def train(c: Config):
    model = c.model_object
    data = c.data_object

    _train(c, data, model)


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
        precision=ct.precision,
    )
    trainer.fit(model, datamodule=datamodule, ckpt_path=ct.ckpt_path)

    if isinstance(trainer_logger, NeptuneLogger):
        trainer_logger.log_model_summary(model=model, max_depth=-1)
        trainer_logger.log_hyperparams(c.model_dump())
