import lightning as L
from torch.utils.data import DataLoader

from guitar_set.dataset import GuitarSetDataset, _Transform


class GuitarSetDataModule(L.LightningDataModule):
    def __init__(
        self,
        *,
        annotation_dir: str,
        audio_dir: str,
        batch_size: int,
        transform: _Transform | None = None,
    ):
        super().__init__()

        self.annotation_dir = annotation_dir
        self.audio_dir = audio_dir
        self.transform = transform
        self.batch_size = batch_size

    def setup(self, stage=None):
        dataset = GuitarSetDataset(
            annotation_dir=self.annotation_dir,
            audio_dir=self.audio_dir,
            sample_rate=22050,
            transform=self.transform,
        )

        self.train_dataset = dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )
