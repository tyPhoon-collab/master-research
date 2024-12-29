metadata_dir = "E:/Dataset/FMA/fma_metadata"
audio_dir = "E:/Dataset/FMA/fma_small"
sample_rate = 22050


def test_dataset():
    import torch

    from music_controlnet.module.data.dataset import FMADataset

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

    from music_controlnet.module.data.datamodule import FMAMelSpectrogramDataModule
    from music_controlnet.script.config import MelConfig

    c = MelConfig()

    datamodule = FMAMelSpectrogramDataModule(
        metadata_dir=metadata_dir,
        audio_dir=audio_dir,
        mel_config=c,
        batch_size=16,
    )

    datamodule.setup("fit")

    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()

    assert len(train_dataloader) > 0
    assert len(val_dataloader) > 0

    mel, genres = next(iter(train_dataloader))

    assert mel.ndim == 4  # batch, channel, height, width
    assert mel.size(0) == 16  # batch size
    assert mel.size(1) == 1  # channel is mono
    assert mel.size(2) == c.n_mels  # height is mel bins
    assert mel.size(3) == c.fixed_length  # width is time steps. trimmed value
    assert isinstance(genres, torch.Tensor)


def test_unet():
    import torch

    from music_controlnet.module.model.unet import UNet
    from music_controlnet.pipeline import UNetDiffusionPipeline
    from music_controlnet.utils import auto_device

    device = auto_device()

    model = UNet().to(device)

    pipeline = UNetDiffusionPipeline(model)

    sample = pipeline(timesteps=1)

    assert isinstance(sample, torch.Tensor)
    assert sample.ndim == 4  # batch, channel, height, width
    assert sample.size(1) == 1  # channel is mono
    assert sample.size(2) == 160  # height is mel bins
    assert sample.size(3) == 2560  # width is time steps. trimmed to 2560
    assert sample.isnan().sum() == 0
