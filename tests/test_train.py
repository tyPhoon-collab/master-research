from tool.config import DataConfig, TrainConfig

metadata_dir = "E:/Dataset/FMA/fma_metadata"
audio_dir = "E:/Dataset/FMA/fma_small"
sample_rate = 22050


def test_train_unet():
    from tool.cli.train import train_unet
    from tool.config import Config

    c = Config(
        mode="train_unet",
        data=DataConfig(
            metadata_dir=metadata_dir,
            audio_dir=audio_dir,
        ),
        train=TrainConfig(
            batch_size=1,
            fast_dev_run=True,
        ),
    )

    train_unet(c)


def test_train_diffwave():
    from tool.cli.train import train_diffwave
    from tool.config import Config

    c = Config(
        mode="train_diffwave",
        data=DataConfig(
            metadata_dir=metadata_dir,
            audio_dir=audio_dir,
        ),
        train=TrainConfig(
            batch_size=1,
            fast_dev_run=True,
        ),
    )

    train_diffwave(c)
