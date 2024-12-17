from collections.abc import Callable

import torch

TimestepCallbackType = Callable[[int, torch.Tensor], None]
