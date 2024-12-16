from neptune.types import File

from src.pipeline import UNetDiffusionPipeline


class UNetLogger:
    def __init__(self, model):
        super().__init__()

        self.model = model
        self.pipeline = UNetDiffusionPipeline(self.model, self.model.scheduler)

        self.epoch_total_loss = 0
        self.epoch_step_count = 0

    @property
    def log(self):
        return self.model.log

    @property
    def logger(self):
        return self.model.logger

    def training_step(self, loss):
        self.log("train/loss", loss)

        self.epoch_total_loss += loss.item()
        self.epoch_step_count += 1

    def on_train_epoch_end(self) -> None:
        sample = self.pipeline()
        img = sample[0][0].cpu().numpy()

        self.log("train/epoch_loss", self.epoch_total_loss / self.epoch_step_count)

        self.epoch_total_loss = 0
        self.epoch_step_count = 0

        self.logger.experiment["train/sample"].append(File.as_image(img))
