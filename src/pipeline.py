from typing import Any

import torch
from torchaudio.transforms import (
    AmplitudeToDB,
    GriffinLim,
    InverseMelScale,
    MelSpectrogram,
)
from tqdm import tqdm

from src.script.config import MelConfig
from src.transforms import (
    DBToAmplitude,
    Lambda,
    NormalizeMinusOneToOne,
    ToMono,
    TrimOrPad,
)
from src.types_ import TimestepCallback


def _noop_timestep_callback(timestep: int, sample: torch.Tensor) -> None:
    pass


class UNetDiffusionPipeline:
    def __init__(self, model, scheduler: Any | None = None):
        self.model = model
        self.scheduler = scheduler

    @torch.inference_mode()
    def __call__(
        self,
        n_mels: int,
        length: int,
        genres: torch.Tensor | None = None,
        timesteps: int = 1000,
        timestep_callback: TimestepCallback | None = None,
        show_progress: bool = True,
    ) -> torch.Tensor:
        training = self.model.training

        noise_shape = (1, 1, n_mels, length)

        try:
            self.model.eval()

            device = self.model.device

            # genresが指定されていない場合のデフォルト設定
            # Hip-Hop
            genres = genres or torch.tensor([[21]], device=device)

            scheduler = self.scheduler or self.model.scheduler
            scheduler.set_timesteps(timesteps)

            sample = torch.randn(noise_shape, device=device)

            callback = timestep_callback or _noop_timestep_callback

            tqdm_timesteps = tqdm(
                scheduler.timesteps,
                desc="Generating",
                disable=not show_progress,
            )

            for t in tqdm_timesteps:
                noise_pred = self.model(sample, t, genres).sample
                int_t = int(t.item())
                sample = scheduler.step(noise_pred, int_t, sample).prev_sample  # type: ignore

                callback(int_t, sample.clone().cpu())

            return sample

        finally:
            if training:
                self.model.train()


class MelSpectrogramPipeline(torch.nn.Module):
    def __init__(self, config: MelConfig):
        super().__init__()

        c = config

        self.transform = torch.nn.Sequential(
            ToMono(),
            # TrimOrPad(target_length=c.sample_rate * (30 // c.num_segments)),
            MelSpectrogram(
                sample_rate=c.sample_rate,
                n_fft=c.n_fft,
                win_length=c.win_length,
                hop_length=c.hop_length,
                n_mels=c.n_mels,
                power=2.0,
            ),
            AmplitudeToDB(
                stype="power",
                top_db=c.top_db,
            ),
            TrimOrPad(target_length=c.fixed_length),
            NormalizeMinusOneToOne(),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.transform(waveform)


class InverseMelSpectrogramPipeline(torch.nn.Module):
    def __init__(self, config: MelConfig):
        super().__init__()

        c = config

        self.transform = torch.nn.Sequential(
            Lambda(lambda mel: mel / 2 * c.top_db),
            DBToAmplitude(),
            InverseMelScale(
                n_stft=c.n_stft,
                n_mels=c.n_mels,
                sample_rate=c.sample_rate,
            ),
            GriffinLim(
                n_fft=c.n_fft,
                win_length=c.win_length,
                hop_length=c.hop_length,
                power=2.0,
            ),
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.transform(mel)
