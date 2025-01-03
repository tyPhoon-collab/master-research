from tool.config import Config, TrainConfig

metadata_dir = "E:/Dataset/FMA/fma_metadata"
audio_dir = "E:/Dataset/FMA/fma_small"
sample_rate = 22050


def test_dataset():
    import torch

    from fma.dataset import FMADataset

    dataset = FMADataset(
        metadata_dir=metadata_dir,
        audio_dir=audio_dir,
        sample_rate=sample_rate,
        transform=lambda x: {
            "waveform": x,
            "mel": x,  # dummy
        },
    )

    assert len(dataset) > 0

    batch = dataset[0]

    waveform = batch["waveform"]
    mel = batch["mel"]
    genres = batch["genres"]

    assert isinstance(waveform, torch.Tensor)
    assert isinstance(mel, torch.Tensor)
    assert isinstance(genres, torch.Tensor)


def test_datamodule():
    import torch

    from tool.factory import build_unet_datamodule

    c = Config(
        train=TrainConfig(
            metadata_dir=metadata_dir,
            audio_dir=audio_dir,
            batch_size=16,
        )
    )

    datamodule = build_unet_datamodule(c)
    datamodule.setup("fit")

    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()

    assert len(train_dataloader) > 0
    assert len(val_dataloader) > 0

    batch = next(iter(train_dataloader))

    mel = batch["mel"]
    genres = batch["genres"]

    assert mel.ndim == 4  # batch, channel, height, width
    assert mel.size(0) == 16  # batch size
    assert mel.size(1) == 1  # channel is mono
    assert mel.size(2) == c.mel.n_mels  # height is mel bins
    assert mel.size(3) == c.mel.fixed_length  # width is time steps. trimmed value
    assert isinstance(genres, torch.Tensor)


def test_collate_fn():
    import torch

    from fma.dataset import collate_fn
    from fma.metadata import PADDING_INDEX

    batch = [
        {"genres": torch.tensor([1, 2]), "mel": torch.randn(1, 160, 2560)},
        {"genres": torch.tensor([4, 5, 6]), "mel": torch.randn(1, 160, 2560)},
    ]

    collated = collate_fn(batch)

    assert collated.keys() == {"mel", "genres"}
    assert collated["mel"].ndim == 4  # batch, channel, height, width
    assert collated["mel"].size(0) == 2  # batch size
    assert collated["genres"].ndim == 2  # batch, length
    assert collated["genres"].size(0) == 2  # batch size
    assert torch.equal(collated["genres"][0], torch.tensor([1, 2, PADDING_INDEX]))
    assert torch.equal(collated["genres"][1], torch.tensor([4, 5, 6]))
