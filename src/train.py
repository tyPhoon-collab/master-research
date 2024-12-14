def train_unet():
    import lightning as L

    from module.data.datamodule import FMAMelSpectrogramDataModule
    from module.model.unet import UNet

    datamodule = FMAMelSpectrogramDataModule(
        metadata_dir="./data/FMA/fma_metadata",
        audio_dir="./data/FMA/fma_small",
        batch_size=2,
        sample_rate=22050,
    )
    model = UNet()

    trainer = L.Trainer(
        max_epochs=1,
        # fast_dev_run=True,
    )
    trainer.fit(model, datamodule=datamodule)
