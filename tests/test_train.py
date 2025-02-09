from tests.utils import load_data_config
from tool.config import Config, TrainConfig


def test_train_unet():
    from fma.metadata import NUM_GENRES
    from tool.cli.train import train

    c = Config(
        data=load_data_config(),
        train=TrainConfig(
            fast_dev_run=True,
        ),
        model={
            "_target_": "music_controlnet.module.unet.UNetLightning",
            "num_class_embeds": NUM_GENRES,
        },
    )

    train(c)


def test_train_diffwave():
    from tool.cli.train import train

    c = Config(
        data=load_data_config(),
        train=TrainConfig(
            fast_dev_run=True,
        ),
        model={
            "_target_": "vocoder.module.diffwave.DiffWaveLightning",
            "n_mels": 128,
        },
    )

    train(c)
