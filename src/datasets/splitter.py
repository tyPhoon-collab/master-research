import torch


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
