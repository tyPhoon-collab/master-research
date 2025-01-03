from logging import getLogger

import torch
from tqdm import tqdm

from fma.dataset import FMADataset
from tool.config import Config
from tool.pipeline import MelSpectrogramPipeline, WaveformPipeline

logger = getLogger(__name__)


def doctor(c: Config):
    logger.info("Starting doctor checks...")

    pipe_waveform = WaveformPipeline(c.mel)
    pipe_mel = MelSpectrogramPipeline(c.mel)

    dataset = FMADataset(
        metadata_dir=c.train.metadata_dir,
        audio_dir=c.train.audio_dir,
        sample_rate=c.mel.sr,
        transform=lambda x: {
            "x": x,
            "waveform": pipe_waveform(x),
            "mel": pipe_mel(x),
        },
        num_segments=c.mel.num_segments,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
    )

    num_nan_x = 0
    num_nan_waveform = 0
    num_nan_mel = 0
    num_nan_genres = 0
    num_zero_x = 0
    num_zero_waveform = 0

    logger.info(f"Checking {len(dataset)} files for issues...")

    for batch in tqdm(dataloader, desc="Validating dataset", unit="batch"):
        x = batch["x"]
        waveform = batch["waveform"]
        mel = batch["mel"]
        genres = batch["genres"]
        audio_path = batch["audio_path"][0]

        def log_issue(issue, data_name):
            logger.warning(f"{issue} in {data_name}. Audio path: {audio_path}")

        if torch.isnan(x).any():
            num_nan_x += 1
            log_issue("NaN found", "x")

        if torch.isnan(waveform).any():
            num_nan_waveform += 1
            log_issue("NaN found", "waveform")

        if torch.isnan(mel).any():
            num_nan_mel += 1
            log_issue("NaN found", "mel")

        if torch.isnan(genres).any():
            num_nan_genres += 1
            log_issue("NaN found", "genres")

        if torch.all(x == 0):
            num_zero_x += 1
            log_issue("All-zero x detected", "x")

        if torch.all(waveform == 0):
            num_zero_waveform += 1
            log_issue("All-zero waveform detected", "waveform")

    logger.info("Doctor checks completed.")
    logger.info("Summary:")
    logger.info(f"  - NaN in x: {num_nan_x}")
    logger.info(f"  - NaN in waveform: {num_nan_waveform}")
    logger.info(f"  - NaN in mel: {num_nan_mel}")
    logger.info(f"  - NaN in genres: {num_nan_genres}")
    logger.info(f"  - All-zero x: {num_zero_x}")
    logger.info(f"  - All-zero waveforms: {num_zero_waveform}")

    if (
        num_nan_x == 0
        and num_nan_waveform == 0
        and num_nan_mel == 0
        and num_nan_genres == 0
        and num_zero_x == 0
        and num_zero_waveform == 0
    ):
        logger.info("No issues detected in the dataset.")
    else:
        logger.warning(
            "Some issues were found in the dataset. Please check the logs for details."
        )
