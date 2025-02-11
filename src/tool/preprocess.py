import os
import shutil
from abc import ABC, abstractmethod

import numpy as np
import soundfile as sf
import torchaudio


class AudioSplitterStrategy(ABC):
    @abstractmethod
    def split(self, y: np.ndarray, sr: int) -> list[np.ndarray]:
        pass


class FixedLengthSplitter(AudioSplitterStrategy):
    def __init__(self, length: float, padding: bool = True):
        self.length = length
        self.padding = padding

    def split(self, y: np.ndarray, sr: int) -> list[np.ndarray]:
        segment_samples = int(self.length * sr)
        total_samples = len(y)
        segments: list[np.ndarray] = []
        for start in range(0, total_samples, segment_samples):
            segment = y[start : start + segment_samples]
            if len(segment) < segment_samples:
                if self.padding:
                    pad_width = segment_samples - len(segment)
                    segment = np.concatenate([segment, np.zeros(pad_width)])
                else:
                    break
            segments.append(segment)
        return segments


class GuitarSetPreprocessor:
    def __init__(
        self,
        splitter: AudioSplitterStrategy,
        audio_dir: str,
        output_dir: str,
        clear_output_dir: bool = True,
    ):
        self.splitter = splitter
        self.audio_dir = audio_dir
        self.output_dir = output_dir
        self.clear_output_dir = clear_output_dir

    def __call__(self) -> None:
        self._prepare_output_dir()
        for audio_path in self._get_audio_paths():
            y, sr = self._load_audio(audio_path)
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            segments = self.splitter.split(y, sr)
            self._save_segments(base_name, segments, sr)

    def _prepare_output_dir(self) -> None:
        if self.clear_output_dir and os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

    def _get_audio_paths(self) -> list[str]:
        return [
            os.path.join(self.audio_dir, f)
            for f in os.listdir(self.audio_dir)
            if "comp" in f
        ]

    def _load_audio(self, audio_path: str) -> tuple[np.ndarray, int]:
        waveform, sr = torchaudio.load(audio_path)
        return waveform.detach().numpy()[0], sr

    def _save_segments(
        self, base_name: str, segments: list[np.ndarray], sr: int
    ) -> None:
        for idx, segment in enumerate(segments):
            output_file = os.path.join(
                self.output_dir, f"{base_name}_segment_{idx:03d}.wav"
            )
            sf.write(output_file, segment, sr)
