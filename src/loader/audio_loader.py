import os
from abc import ABC, abstractmethod

import librosa
import torch
import torchaudio

AudioLoadResult = tuple[torch.Tensor, int]


class AudioLoader(ABC):
    @abstractmethod
    def load(self, audio_path: str) -> AudioLoadResult: ...


class TorchAudioLoader(AudioLoader):
    def load(self, audio_path: str) -> AudioLoadResult:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            waveform, sr = torchaudio.load(audio_path)
        except RuntimeError:
            raise RuntimeError(f"Failed to load audio file: {audio_path}")
        return waveform, sr


class LibrosaAudioLoader(AudioLoader):
    def load(self, audio_path: str) -> AudioLoadResult:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            waveform, sr = librosa.load(audio_path, sr=None, mono=False)
        except RuntimeError:
            raise RuntimeError(f"Failed to load audio file: {audio_path}")
        return torch.tensor(waveform), int(sr)
