import librosa
import numpy as np
import torch

from .functions import fixed_time_axis_length
from .util import Clamp, Mono, Scale, TrimOrPad


class Reassigned(torch.nn.Module):
    def __init__(
        self,
        audio_duration: int,
        n_segments: int = 1,
        sample_rate: int = 22050,
        interval: float | None = None,
        hop_length: int = 256,
        n_fft: int = 1024,
        win_length: int = 1024,
        freq_bins: np.ndarray | None = None,
        density: bool = False,
        # top_db: int = 80,
        reassign_times: bool = False,
    ):
        super().__init__()

        fixed_length = fixed_time_axis_length(
            audio_duration=audio_duration,
            n_segments=n_segments,
            sample_rate=sample_rate,
            hop_length=hop_length,
        )

        self.transform = torch.nn.Sequential(
            Mono(),
            Clamp.one(),
            _Reassigned(
                audio_duration,
                sample_rate,
                interval,
                n_fft,
                win_length,
                hop_length,
                freq_bins,
                density,
                reassign_times,
            ),
            TrimOrPad(target_length=fixed_length, mode="replicate"),
            # AmplitudeToDB(stype="power", top_db=top_db),
            Scale.one(),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.transform(waveform)


class _Reassigned(torch.nn.Module):
    def __init__(
        self,
        audio_duration: float,
        sample_rate: int,
        interval: float | None,
        n_fft: int,
        win_length: int,
        hop_length: int,
        freq_bins: np.ndarray | None,
        density: bool,
        reassign_times: bool,
    ):
        super().__init__()
        self.audio_duration = audio_duration
        self.sample_rate = sample_rate
        self.interval = interval if interval is not None else hop_length / sample_rate

        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.density = density
        self.reassign_times = reassign_times

        self.time_bins = np.arange(0, self.audio_duration, self.interval)
        self.freq_bins = self.calculate_freq_bin() if freq_bins is None else freq_bins
        self.bins = [self.time_bins, self.freq_bins]

    def calculate_freq_bin(self) -> np.ndarray:
        """
        ピアノの88鍵の音域の周波数ビンを作成
        ビン幅は 前の音と対象の音の中点 ~ 対象の音と次の音の中点
        """
        # 音テーブルの作成。A0~C8
        # 音域の参考サイト: https://tomari.org/main/java/oto.html
        ratio = 1.059463094
        lowest_hz = 27.5
        hz_list = [lowest_hz * ratio**i for i in range(89)]
        bins = [(x + y) / 2 for x, y in zip([lowest_hz / ratio, *hz_list], hz_list)]
        return np.array(bins)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        y = waveform.detach().cpu().numpy().astype(np.float32)

        freqs, times, mags = librosa.reassigned_spectrogram(
            y,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            fill_nan=True,
            reassign_times=self.reassign_times,
        )

        h, _, _ = np.histogram2d(
            times.flatten(),
            freqs.flatten(),
            bins=self.bins,
            weights=mags.flatten(),
            density=self.density,
        )

        h_tensor = torch.tensor(h, dtype=waveform.dtype)
        return h_tensor.T
