from fma.datamodule import FMADataModule
from tool.config import Config
from tool.pipeline import MelSpectrogramPipeline, WaveformPipeline


def build_unet_datamodule(config: Config) -> FMADataModule:
    mel_pipeline = MelSpectrogramPipeline(config.mel)

    return FMADataModule(
        metadata_dir=config.data.metadata_dir,
        audio_dir=config.data.audio_dir,
        sample_rate=config.mel.sr,
        num_segments=config.mel.num_segments,
        transform=lambda x: {
            "mel": mel_pipeline(x),
        },
        batch_size=config.train.batch_size,
    )


def build_diffwave_datamodule(config: Config) -> FMADataModule:
    waveform_pipeline = WaveformPipeline(config.mel)
    mel_pipeline = MelSpectrogramPipeline(config.mel)

    return FMADataModule(
        metadata_dir=config.data.metadata_dir,
        audio_dir=config.data.audio_dir,
        sample_rate=config.mel.sr,
        num_segments=config.mel.num_segments,
        transform=lambda x: {
            "waveform": waveform_pipeline(x),
            "mel": mel_pipeline(x),
        },
        batch_size=config.train.batch_size,
    )
