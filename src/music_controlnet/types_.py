from collections.abc import Callable

import torch

Genres = torch.Tensor
FMADatasetReturn = tuple[torch.Tensor, Genres]
Transform = Callable[[torch.Tensor], torch.Tensor]
