from collections.abc import Callable

import torch

Genres = torch.Tensor
FMADatasetReturn = dict[str, torch.Tensor]
Transform = Callable[[torch.Tensor], dict[str, torch.Tensor]]
