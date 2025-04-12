import librosa
import numpy as np
import torch


class Reassigned(torch.nn.Module):
    def __init__(
        self,
        audio_duration: float,
        sample_rate: int = 22050,
        interval: float = 0.01,
        freq_bins: np.ndarray | None = None,
        density: bool = False,
        **reassigned_kwargs,
    ):
        super().__init__()
        self.audio_duration = audio_duration
        self.sample_rate = sample_rate
        self.interval = interval
        self.time_bins = np.arange(0, audio_duration, interval)
        self.freq_bins = self.calculate_freq_bin() if freq_bins is None else freq_bins
        self.bins = [self.time_bins, self.freq_bins]
        self.density = density
        self.reassigned_kwargs = reassigned_kwargs

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
            y, sr=self.sample_rate, fill_nan=True, **self.reassigned_kwargs
        )

        h, _, _ = np.histogram2d(
            times.flatten(),
            freqs.flatten(),
            bins=self.bins,
            weights=mags.flatten(),
            density=self.density,
        )

        # 結果の numpy 配列を torch.Tensor に変換して返す
        h_tensor = torch.tensor(h, dtype=waveform.dtype)
        return h_tensor
