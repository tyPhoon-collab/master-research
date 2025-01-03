from collections.abc import Callable

import torch

Genres = torch.Tensor
FMADatasetReturn = dict[str, torch.Tensor | str]
Transform = Callable[[torch.Tensor], FMADatasetReturn]
