from cli.config import Config
from tests.utils import load_data_config


def test_fma_dataset():
    import torch

    from datasets.fma.dataset import FMADataset

    c = load_data_config()

    dataset = FMADataset(
        metadata_dir=c["metadata_dir"],
        audio_dir=c["audio_dir"],
        sample_rate=22050,
        transform=lambda x: {
            "waveform": x,
            "mel": x,  # dummy
        },
    )

    assert len(dataset) > 0

    batch = dataset[0]

    waveform = batch["waveform"]
    mel = batch["mel"]
    genre = batch["genre"]

    assert isinstance(waveform, torch.Tensor)
    assert isinstance(mel, torch.Tensor)
    assert isinstance(genre, torch.Tensor)


def test_fma_datamodule():
    import torch

    c = Config(
        data=load_data_config(),
    )

    datamodule = c.data_object
    datamodule.setup("fit")

    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()

    assert len(train_dataloader) > 0
    assert len(val_dataloader) > 0

    batch = next(iter(train_dataloader))

    mel = batch["mel"]
    genre = batch["genre"]

    assert mel.ndim == 4  # batch, channel, height, width
    assert mel.size(0) == 4  # batch size
    assert mel.size(1) == 1  # channel is mono
    assert mel.size(2) == 128  # height is mel bins
    assert mel.size(3) == 864  # width is time steps. trimmed value
    assert isinstance(genre, torch.Tensor)
