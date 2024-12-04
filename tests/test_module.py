import torch

from src.module.data.datamodule import FMAMelSpectrogramDataModule
from src.module.data.dataset import FMADataset

metadata_dir = "E:/Dataset/FMA/fma_metadata"
audio_dir = "E:/Dataset/FMA/fma_small"
sample_rate = 22050


def test_dataset():
    dataset = FMADataset(
        metadata_dir=metadata_dir,
        audio_dir=audio_dir,
        sample_rate=sample_rate,
        transform=None,
    )

    assert len(dataset) > 0

    waveform, genre = dataset[0]

    assert isinstance(waveform, torch.Tensor)
    assert isinstance(genre, str)
    assert genre == "Hip-Hop"


def test_datamodule():
    datamodule = FMAMelSpectrogramDataModule(
        metadata_dir=metadata_dir,
        audio_dir=audio_dir,
        sample_rate=sample_rate,
        batch_size=32,
        val_split=0.2,
    )

    datamodule.setup("fit")

    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()

    assert len(train_dataloader) > 0
    assert len(val_dataloader) > 0

    mel, genre = next(iter(train_dataloader))

    assert mel.ndim == 4  # batch, channel, height, width
    assert isinstance(genre, tuple)
