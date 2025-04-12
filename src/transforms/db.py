from typing import Literal

import torch
import torchaudio


class DBToAmplitude(torch.nn.Module):
    def __init__(self, ttype: Literal["power", "magnitude"] = "power") -> None:
        super().__init__()
        self.ttype = ttype

    def forward(self, db: torch.Tensor) -> torch.Tensor:
        return torchaudio.functional.DB_to_amplitude(
            db,
            ref=1.0,
            power=1 if self.ttype == "power" else 0.5,
        )
