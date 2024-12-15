import sys

sys.path.append("src")

metadata_dir = "E:/Dataset/FMA/fma_metadata"
audio_dir = "E:/Dataset/FMA/fma_small"
sample_rate = 22050


def test_dataset():
    import torch

    from module.data.dataset import FMADataset

    dataset = FMADataset(
        metadata_dir=metadata_dir,
        audio_dir=audio_dir,
        sample_rate=sample_rate,
        transform=None,
    )

    assert len(dataset) > 0

    waveform, genres = dataset[0]

    assert isinstance(waveform, torch.Tensor)
    assert isinstance(genres, torch.Tensor)


def test_datamodule():
    import torch

    from module.data.datamodule import FMAMelSpectrogramDataModule

    datamodule = FMAMelSpectrogramDataModule(
        metadata_dir=metadata_dir,
        audio_dir=audio_dir,
        sample_rate=sample_rate,
        batch_size=16,
    )

    datamodule.setup("fit")

    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()

    assert len(train_dataloader) > 0
    assert len(val_dataloader) > 0

    mel, genres = next(iter(train_dataloader))

    assert mel.ndim == 4  # batch, channel, height, width
    assert mel.size(1) == 1  # channel is mono
    assert mel.size(2) == 160  # height is mel bins
    assert mel.size(3) == 2560  # width is time steps. trimmed to 2560
    assert isinstance(genres, torch.Tensor)
