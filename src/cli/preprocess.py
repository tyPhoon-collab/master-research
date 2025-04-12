from cli.config import Config


def preprocess(cfg: Config):
    preprocessor = cfg.preprocess_object

    preprocessor()
