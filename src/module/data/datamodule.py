from os import PathLike

import lightning as L
from torch.utils.data import DataLoader, random_split
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram
from torchvision.transforms import Compose

from src.module.data.dataset import FMADataset


class FMAMelSpectrogramDataModule(L.LightningDataModule):
    def __init__(
        self,
        *,
        metadata_dir: PathLike | str,
        audio_dir: PathLike | str,
        sample_rate: int,
        batch_size: int,
        val_split: float = 0.2,
    ):
        super().__init__()
        self.metadata_dir = metadata_dir
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.transform = Compose(
            [
                MelSpectrogram(
                    sample_rate=sample_rate,
                    n_fft=2048,
                    win_length=2048,
                    hop_length=256,
                    n_mels=160,
                    power=2.0,
                ),
                AmplitudeToDB(),
                lambda mel: mel[..., :2560],
            ]
        )
        self.val_split = val_split

    def setup(self, stage: str):
        if stage != "fit":
            raise NotImplementedError()

        dataset = FMADataset(
            metadata_dir=self.metadata_dir,
            audio_dir=self.audio_dir,
            sample_rate=self.sample_rate,
            transform=self.transform,
        )

        val_size = int(len(dataset) * self.val_split)
        train_size = len(dataset) - val_size

        self.train_dataset, self.val_dataset = random_split(
            dataset, [train_size, val_size]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=FMADataset.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=FMADataset.collate_fn,
        )
