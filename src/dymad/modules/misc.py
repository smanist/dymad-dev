import torch
import torch.nn as nn


class TakeFirst(nn.Module):
    """
    Pass-through layer that returns the first `m` entries in the last axis.

    Args:
        m (int): Number of entries to take from the last axis.
    """

    def __init__(self, m: int):
        super().__init__()
        assert m > 0, "m must be a positive integer"
        self.m = m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """"""
        return x[..., : self.m] if x.ndim > 1 else x[: self.m]

    def diagnostic_info(self) -> str:
        return f"m: {self.m}"

    def __repr__(self) -> str:
        return f"TakeFirst(m={self.m})"


class TakeFirstGraph(TakeFirst):
    """
    Graph version of TakeFirst.

    Input (..., n_nodes, n_features)
    Output (..., n_nodes*m)
    """

    def forward(
        self, x: torch.Tensor, edge_index, edge_weights, edge_attr, **kwargs
    ) -> torch.Tensor:
        """"""
        out_shape = x.shape[:-2] + (-1,)
        return x[..., : self.m].reshape(*out_shape) if x.ndim > 1 else x[: self.m]

    def __repr__(self) -> str:
        return f"TakeFirstGraph(m={self.m})"
