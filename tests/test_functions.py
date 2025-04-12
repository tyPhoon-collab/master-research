def test_nearest_multiple():
    from transforms.functions import nearest_multiple

    assert nearest_multiple(1, 2) == 2
    assert nearest_multiple(2, 2) == 2
    assert nearest_multiple(3, 2) == 4
    assert nearest_multiple(4, 2) == 4

    assert nearest_multiple(22050 * 30 // 256, 16) == 2576


def test_fixed_mel_length():
    from transforms.functions import fixed_mel_length

    fixed = fixed_mel_length(
        audio_duration=30,
        n_segments=3,
        sample_rate=22050,
        hop_length=256,
    )
    assert fixed == 864
