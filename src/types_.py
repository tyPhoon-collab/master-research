from collections.abc import Callable

import torch

TimestepCallbackType = Callable[[int, torch.Tensor], None]


Genres = torch.Tensor
FMADatasetReturn = tuple[torch.Tensor, Genres]
Transform = Callable[[torch.Tensor], torch.Tensor]
