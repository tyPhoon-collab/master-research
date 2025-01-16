from lightning import Callback


class ScheduleFreeOptimizerCallback(Callback):
    """
    Optimizerのmodeを切り替えるCallback

    `optimizer`, `optimizer_g`, `optimizer_d` などが pl_module に存在し、
    かつそれらが `train`, `eval` メソッドを持っている場合のみ影響を与える。
    """

    def on_train_epoch_start(self, trainer, pl_module):
        self._set_optimizers_mode(pl_module, "train")

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        self._set_optimizers_mode(pl_module, "eval")

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        self._set_optimizers_mode(pl_module, "train")

    def _set_optimizers_mode(self, pl_module, mode):
        """
        pl_module 内の `optimizer`, `optimizer_g`, `optimizer_d` などに対して
        `mode` (train/eval) を適用する。
        """
        optimizer_names = ["optimizer", "optimizer_g", "optimizer_d"]

        for opt_name in optimizer_names:
            optimizer = getattr(pl_module, opt_name, None)
            if optimizer and callable(getattr(optimizer, mode, None)):
                getattr(optimizer, mode)()
