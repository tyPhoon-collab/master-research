import torch


class ToMono(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform
