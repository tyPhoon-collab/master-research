from abc import ABC, abstractmethod
from typing import Any

import torch
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from fma.dataset import FMADataset
from tool.config import Config
from tool.pipeline import MelSpectrogramPipeline, WaveformPipeline

console = Console()


class _Validator(ABC):
    @abstractmethod
    def __call__(self, batch: dict[str, Any]) -> None | str:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class _NaNValidator(_Validator):
    def __init__(self, key: str):
        self._key = key

    def __call__(self, batch: dict[str, Any]) -> None | str:
        if torch.isnan(batch[self._key]).any():
            return f"NaN found in {self._key}"
        return None

    @property
    def name(self) -> str:
        return f"NaN in {self._key}"


class _ZeroValidator(_Validator):
    def __init__(self, key: str):
        self._key = key

    def __call__(self, batch: dict[str, Any]) -> None | str:
        if torch.all(batch[self._key] == 0):
            return f"All-zero detected in {self._key}"
        return None

    @property
    def name(self) -> str:
        return f"All-zero in {self._key}"


class _Doctor:
    def __init__(self, validators: list[_Validator]):
        self.validators = validators
        self.issues_audio_paths = {validator.name: [] for validator in self.validators}

    def __call__(self, batch: dict[str, Any]) -> None:
        audio_path = batch["audio_path"][0]

        for validator in self.validators:
            issue = validator(batch)
            if issue is not None:
                console.print(
                    f"Warning: {validator.name}: {issue}. Audio path: {audio_path}",
                    style="bold red",
                )
                self.issues_audio_paths[validator.name].append(audio_path)

    def report(self):
        console.print("\nValidation Summary:", style="bold cyan")

        table = Table(title="Validation Results", header_style="bold magenta")
        table.add_column("Issue", style="bold")
        table.add_column("Count", justify="right")

        for issue_name, files in self.issues_audio_paths.items():
            table.add_row(issue_name, str(len(files)))

        console.print(table)

        for issue_name, files in self.issues_audio_paths.items():
            if files:
                console.print(f"\nFiles with {issue_name}:", style="bold red")
                for file in files:
                    console.print(f"  - {file}")

        if all(len(files) == 0 for files in self.issues_audio_paths.values()):
            console.print("No issues detected in the dataset.", style="bold green")
        else:
            console.print(
                "Some issues were found. Please check the details above.",
                style="bold yellow",
            )


def doctor(c: Config):
    console.print("Starting dataset validation...", style="bold cyan")

    pipe_waveform = WaveformPipeline(c.mel)
    pipe_mel = MelSpectrogramPipeline(c.mel)

    dataset = FMADataset(
        metadata_dir=c.data.metadata_dir,
        audio_dir=c.data.audio_dir,
        sample_rate=c.mel.sr,
        transform=lambda x: {
            "x": x,
            "waveform": pipe_waveform(x),
            "mel": pipe_mel(x),
        },
        num_segments=c.mel.num_segments,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
    )

    validators = [
        _NaNValidator("x"),
        _NaNValidator("waveform"),
        _NaNValidator("mel"),
        _ZeroValidator("x"),
        _ZeroValidator("waveform"),
    ]

    doctor = _Doctor(validators)

    console.print(
        f"Checking {len(dataset)} files for issues...\n", style="bold magenta"
    )

    for batch in tqdm(dataloader, desc="Validating dataset", unit="batch"):
        doctor(batch)

    doctor.report()
