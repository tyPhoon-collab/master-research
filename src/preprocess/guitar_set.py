import os
import shutil

import librosa
import numpy as np
import soundfile as sf
import torchaudio

from preprocess.splitter import AudioSplitter


def _get_comp_audio_paths(audio_dir: str) -> list[str]:
    return [
        os.path.join(audio_dir, f)
        for f in os.listdir(audio_dir)
        if "comp" in f and os.path.isfile(os.path.join(audio_dir, f))
    ]


class GuitarSetPreprocessor:
    def __init__(
        self,
        splitter: AudioSplitter,
        audio_dir: str,
        output_dir: str,
        with_clear_output_dir: bool = True,
        semitone_shifts: list[int] | None = None,
    ):
        self.splitter = splitter
        self.audio_dir = audio_dir
        self.output_dir = output_dir
        self.with_clear_output_dir = with_clear_output_dir
        self.semitone_shifts = semitone_shifts if semitone_shifts is not None else [0]

    def __call__(self) -> None:
        self._prepare_output_dir()
        for audio_path in _get_comp_audio_paths(self.audio_dir):
            y, sr = self._load_audio(audio_path)
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            segments = self.splitter.split(y, sr)
            self._save_shifted_segments(base_name, segments, sr)

    def _prepare_output_dir(self) -> None:
        if self.with_clear_output_dir and os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

    def _load_audio(self, audio_path: str) -> tuple[np.ndarray, int]:
        waveform, sr = torchaudio.load(audio_path)

        # GuitarSetはモノラル音源のため、一次元で管理する
        return waveform.detach().numpy()[0], sr

    def _save_shifted_segments(
        self, base_name: str, segments: list[np.ndarray], sr: int
    ) -> None:
        for idx, segment in enumerate(segments):
            for shift in self.semitone_shifts:
                shifted_segment = self._pitch_shift(segment, sr, shift)
                output_file = os.path.join(
                    self.output_dir,
                    f"{base_name}_segment_{idx:03d}_shift_{shift:+03d}.wav",
                )
                sf.write(output_file, shifted_segment, sr)

    def _pitch_shift(self, segment: np.ndarray, sr: int, semitones: int) -> np.ndarray:
        return librosa.effects.pitch_shift(segment, sr=sr, n_steps=semitones)
