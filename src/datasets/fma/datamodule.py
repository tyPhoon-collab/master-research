import lightning as L
from torch.utils.data import DataLoader, random_split

from .dataset import FMADataset, _Transform


class FMADataModule(L.LightningDataModule):
    def __init__(
        self,
        *,
        metadata_dir: str,
        audio_dir: str,
        sample_rate: int,
        n_segments: int,
        transform: _Transform | None,
        batch_size: int,
        val_size: int = 1,
    ):
        super().__init__()
        self.metadata_dir = metadata_dir
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.n_segments = n_segments
        self.batch_size = batch_size
        self.transform = transform
        self.val_size = val_size

    def setup(self, stage: str):
        dataset = FMADataset(
            metadata_dir=self.metadata_dir,
            audio_dir=self.audio_dir,
            sample_rate=self.sample_rate,
            n_segments=self.n_segments,
            transform=self.transform,
        )

        train_size = len(dataset) - self.val_size

        self.train_dataset, self.val_dataset = random_split(
            dataset, [train_size, self.val_size]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )
