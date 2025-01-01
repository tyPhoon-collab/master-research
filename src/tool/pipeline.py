import torch
from torchaudio.transforms import (
    AmplitudeToDB,
    GriffinLim,
    InverseMelScale,
    MelSpectrogram,
)

from tool.config import MelConfig
from tool.transforms import (
    Clamp,
    DBToAmplitude,
    Lambda,
    Scale,
    ToMono,
    TrimOrPad,
)


class MelSpectrogramPipeline(torch.nn.Module):
    def __init__(self, config: MelConfig):
        super().__init__()

        c = config

        self.transform = torch.nn.Sequential(
            ToMono(),
            Clamp.one(),
            MelSpectrogram(
                sample_rate=c.sample_rate,
                n_fft=c.n_fft,
                win_length=c.win_length,
                hop_length=c.hop_length,
                n_mels=c.n_mels,
                power=2.0,
                normalized=True,
            ),
            TrimOrPad(target_length=c.fixed_length),
            AmplitudeToDB(
                stype="power",
                top_db=c.top_db,
            ),
            Scale.one(),
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
                n_stft=c.n_fft // 2 + 1,
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