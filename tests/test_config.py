def test_mel_config():
    from tool.config import MelConfig

    c = MelConfig(
        audio_duration=30,
        n_segments=3,
        sr=22050,
        hop_length=256,
    )

    # 22050 * 30 // 3 // 256 = 861.328125
    # 861.328125 -> 864 # nearest 16 multiple
    assert c.fixed_mel_length == 864


def test_config_model_dump():
    from tool.config import Config

    c = Config()
    dumped = c.model_dump()

    assert dumped["mode"] == "train_unet"
