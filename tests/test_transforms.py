def test_trim_or_pad():
    import torch

    from tool.transforms import TrimOrPad

    trim_or_pad = TrimOrPad(100)

    assert trim_or_pad(torch.randn(1, 200)).shape == (1, 100)
    assert trim_or_pad(torch.randn(1, 100)).shape == (1, 100)
    assert trim_or_pad(torch.randn(1, 50)).shape == (1, 100)

    assert trim_or_pad(torch.randn(1, 1, 200)).shape == (1, 1, 100)
    assert trim_or_pad(torch.randn(1, 1, 100)).shape == (1, 1, 100)
    assert trim_or_pad(torch.randn(1, 1, 50)).shape == (1, 1, 100)


def test_inverse_mel():
    import torch

    from tool.transforms import InverseMel

    n_mels = 128
    hop_length = 256

    inverse_mel = InverseMel(
        sample_rate=22050,
        n_fft=1024,
        win_length=1024,
        hop_length=hop_length,
        n_mels=n_mels,
        top_db=80,
    )

    mel = torch.rand(1, n_mels, 100, dtype=torch.float32)

    wave: torch.Tensor = inverse_mel(mel)

    assert wave.shape == (1, 100 * hop_length)
    assert wave.nansum() != 0
