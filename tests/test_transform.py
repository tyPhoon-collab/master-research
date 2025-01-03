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
