def test_mel_config():
    from tool.config import MelConfig

    c = MelConfig()

    assert c.fixed_mel_length == 864
