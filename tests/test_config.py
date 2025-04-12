def test_config_model_dump():
    """
    For hyperparameter logging, we need to dump the model configuration.
    """
    from cli.config import Config

    c = Config()
    dumped = c.model_dump()

    assert dumped["mode"] == "train"
