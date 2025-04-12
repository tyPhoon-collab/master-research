from cli.config import Config, TrainConfig
from tests.utils import load_data_config


def test_train_unet():
    from cli.train import train
    from datasets.fma.metadata import NUM_GENRES

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
    from cli.train import train

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
