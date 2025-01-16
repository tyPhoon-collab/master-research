def test_unet():
    import torch

    from music_controlnet.module.unet import UNetLightning

    device = "cuda"
    model = UNetLightning().to(device)

    noise = model(
        torch.randn(1, 1, 160, 2560, device=device),
        torch.randint(0, 999, (1,), device=device),
        torch.tensor([[21]], device=device),
    ).sample

    assert isinstance(noise, torch.Tensor)
    assert noise.ndim == 4  # batch, channel, height, width
    assert noise.size(1) == 1  # channel is mono
    assert noise.size(2) == 160  # height is mel bins
    assert noise.size(3) == 2560  # width is time steps. trimmed to 2560
    assert noise.isnan().sum() == 0


def test_unet_generate():
    import torch

    from music_controlnet.module.unet import UNetLightning

    device = "cuda"
    model = UNetLightning().to(device)

    sample = model.generate(160, 2560, timesteps=1)

    assert isinstance(sample, torch.Tensor)
    assert sample.ndim == 4  # batch, channel, height, width
    assert sample.size(1) == 1  # channel is mono
    assert sample.size(2) == 160  # height is mel bins
    assert sample.size(3) == 2560  # width is time steps. trimmed to 2560
    assert sample.isnan().sum() == 0


def test_diffwave():
    import torch

    from vocoder.module.diffwave import DiffWaveLightning

    device = "cuda"
    model = DiffWaveLightning(n_mels=128).to(device)

    audio = model(
        torch.randn(1, 1, 432 * 256, device=device),
        torch.randint(0, 49, (1,), device=device),
        torch.randn(1, 1, 128, 432, device=device),
    )

    assert isinstance(audio, torch.Tensor)


def test_diffwave_generate():
    import torch

    from vocoder.module.diffwave import DiffWaveLightning

    device = "cuda"
    model = DiffWaveLightning(n_mels=128).to(device)

    audio = model.generate(
        torch.randn(1, 1, 128, 432, device=device),
    )

    assert isinstance(audio, torch.Tensor)


def test_music_hifi():
    import torch

    from vocoder.module.music_hifi import MusicHiFiLightning

    device = "cuda"
    n_mels = 128
    model = MusicHiFiLightning(
        n_mels=n_mels,
    ).to(device)

    waveform = model(
        torch.randn(1, n_mels, 864, device=device),
    )

    assert isinstance(waveform, torch.Tensor)
