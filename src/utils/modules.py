import torch
import torch.nn as nn

class TakeFirst(nn.Module):
    """
    Pass-through layer that returns the first `m` entries in the last axis.

    Examples
    --------
    >>> sel = TakeFirst(m=2)
    >>> a = torch.randn(3, 4)
    >>> out = sel(a)
    >>> out.shape
    (3, 2)
    """
    def __init__(self, m: int):
        super().__init__()
        assert m > 0, "m must be a positive integer"
        self.m = m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., :self.m] if x.ndim > 1 else x[:self.m]
