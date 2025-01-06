from lightning import Callback


class ScheduleFreeOptimizerCallback(Callback):
    """
    Optimizerのmodeを切り替えるCallback

    self.optimizerが存在し、かつ
    self.optimizerがtrain, evalメソッドを持っている場合のみ副作用する
    """

    def on_train_epoch_start(self, trainer, pl_module):
        self._set_optimizer_mode(pl_module, "train")

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        self._set_optimizer_mode(pl_module, "eval")

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        self._set_optimizer_mode(pl_module, "train")

    def _set_optimizer_mode(self, pl_module, mode):
        if hasattr(pl_module, "optimizer") and callable(
            getattr(pl_module.optimizer, mode, None)
        ):
            getattr(pl_module.optimizer, mode)()
