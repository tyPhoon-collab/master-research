from omegaconf import OmegaConf


def load_data_config(path: str = "conf/data/fma_small.yaml") -> dict:
    dict_config = OmegaConf.load(path)
    return dict_config  # type: ignore
