import torch
from torchaudio.transforms import AmplitudeToDB, GriffinLim, Spectrogram

from .db import DBToAmplitude
from .functions import fixed_time_axis_length  # mel用だが長さ計算に使える
from .util import Clamp, Lambda, Mono, Scale, TrimOrPad


class STFT(torch.nn.Module):
    def __init__(
        self,
        audio_duration: int,
        n_segments: int = 1,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        win_length: int = 1024,
        hop_length: int = 256,
        power: float = 2.0,
        top_db: int = 80,
        **kwargs,
    ):
        super().__init__()

        if kwargs:
            print(f"Unused kwargs provided: {kwargs}")

        fixed_length = fixed_time_axis_length(
            audio_duration=audio_duration,
            n_segments=n_segments,
            sample_rate=sample_rate,
            hop_length=hop_length,
        )

        self.transform = torch.nn.Sequential(
            Mono(),
            Clamp.one(),
            Spectrogram(
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                power=power,
                normalized=True,
            ),
            TrimOrPad(target_length=fixed_length, mode="replicate"),
            AmplitudeToDB(stype="power", top_db=top_db),
            Scale.one(),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.transform(waveform)


class InverseSTFT(torch.nn.Module):
    def __init__(
        self,
        n_fft: int = 1024,
        win_length: int = 1024,
        hop_length: int = 256,
        power: float = 2.0,
        top_db: int = 80,
    ):
        super().__init__()

        self.transform = torch.nn.Sequential(
            Lambda(lambda x: x / 2 * top_db),  # dB → スケール復元
            DBToAmplitude(),  # dB → amplitude
            GriffinLim(
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                power=power,
            ),
        )

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        return self.transform(spec)
