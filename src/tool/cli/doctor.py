from abc import ABC, abstractmethod
from typing import Any

import lightning as L
import torch
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from tool.config import Config

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
        if self._key not in batch:
            return None

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
        if self._key not in batch:
            return None

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

    datamodule: L.LightningDataModule = c.data_object
    datamodule.setup("fit")

    dataloader = datamodule.train_dataloader()

    validators = [
        _NaNValidator("waveform"),
        _NaNValidator("mel"),
        _ZeroValidator("waveform"),
    ]

    doctor = _Doctor(validators)

    console.print(
        f"Checking {len(dataloader)} files for issues...\n", style="bold magenta"
    )

    for batch in tqdm(dataloader, desc="Validating dataset", unit="batch"):
        doctor(batch)

    doctor.report()
