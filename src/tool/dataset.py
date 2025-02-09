import torch
import torchaudio


class Resampler:
    def __init__(self, sample_rate: int):
        self.target_sample_rate = sample_rate

        # for instance cache
        self.resamples: dict[int, torchaudio.transforms.Resample] = {}

    def __call__(self, waveform: torch.Tensor, source_sample_rate: int) -> torch.Tensor:
        if source_sample_rate != self.target_sample_rate:
            resample = self.get_resample(source_sample_rate)
            waveform = resample(waveform)
        return waveform

    def get_resample(self, source_sample_rate: int):
        resample = self.resamples.setdefault(
            source_sample_rate,
            torchaudio.transforms.Resample(
                source_sample_rate,
                self.target_sample_rate,
            ),
        )

        return resample


class Splitter:
    def __init__(self, n_segments: int):
        self.n_segments = n_segments

    def get_id_index(self, id: int) -> int:
        return id // self.n_segments

    def __call__(self, waveform: torch.Tensor, index: int) -> torch.Tensor:
        segment_index = index % self.n_segments

        length = waveform.size(-1) // self.n_segments

        start = segment_index * length
        end = start + length

        return waveform[..., start:end]
