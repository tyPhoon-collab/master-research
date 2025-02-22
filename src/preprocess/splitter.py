from abc import ABC, abstractmethod

import numpy as np


class AudioSplitter(ABC):
    @abstractmethod
    def split(self, y: np.ndarray, sr: int) -> list[np.ndarray]:
        pass


class FixedLengthSplitter(AudioSplitter):
    def __init__(self, length: float, padding: bool = True):
        self.length = length
        self.padding = padding

    def split(self, y: np.ndarray, sr: int) -> list[np.ndarray]:
        segment_samples = int(self.length * sr)
        total_samples = y.shape[0] if y.ndim == 1 else y.shape[-1]
        num_segments = (total_samples + segment_samples - 1) // segment_samples
        segments = []
        for i in range(num_segments):
            start = i * segment_samples
            end = min(start + segment_samples, total_samples)
            segment = y[start:end] if y.ndim == 1 else y[..., start:end]

            if self.padding and segment.shape[-1] < segment_samples:
                if y.ndim == 1:
                    segment = np.pad(segment, (0, segment_samples - segment.shape[-1]))
                elif y.ndim == 2:
                    segment = np.pad(
                        segment, ((0, 0), (0, segment_samples - segment.shape[-1]))
                    )
                else:
                    raise ValueError(f"Expected y.ndim to be 1 or 2, but got {y.ndim}")

            segments.append(segment)
        return segments
