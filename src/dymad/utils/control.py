from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import torch
import torch.nn as nn
from scipy.interpolate import interp1d


class ControlInterpolator(nn.Module):
    """
    Interpolates the sampled control signal u(t_k) when the ODE solver
    requests u(t_query).

    Args:
        t (torch.Tensor): 1-D tensor of shape (N,). Sampling times (must be ascending).
        u (torch.Tensor): Tensor of shape (..., N, m). Control samples, m inputs per step.
        axis (int): Axis of `u` that corresponds to time. Default is -2.
        order (str): Interpolation mode. One of {'none', 'zoh', 'linear', 'cubic', etc}.

    Note:
        Not to be confused with `dymad.utils.sampling._build_interpolant`,
        which is for data generation, esp. with Numpy.
        `ControlInterpolator` is meant to be used in a Torch setting.
    """

    t: torch.Tensor
    u: torch.Tensor | None

    def __init__(
        self,
        t: torch.Tensor,
        u: torch.Tensor | None,
        axis: int = -2,
        order: str = "linear",
    ) -> None:
        super().__init__()

        if u is not None:
            assert u.ndim >= 2, "Control signal must have at least 2 dimensions"

        self.axis = axis
        self.order = order.lower()
        self.register_buffer("t", t)
        self.register_buffer("u", u)
        self._interp: Callable[[torch.Tensor], torch.Tensor | None]

        if u is None:
            self._interp = self._interp_none
        elif self.order == "zoh":
            self._interp = self._interp_0
        elif self.order == "linear":
            self._interp = self._interp_1
        else:
            # Assuming option for 'scipy' interpolation
            self._cpu_t = t.detach().cpu().numpy()
            self._cpu_u = u.detach().cpu().numpy()
            self._spl = interp1d(
                self._cpu_t,
                self._cpu_u,
                kind=order,
                axis=axis,
                fill_value=cast(Any, "extrapolate"),
                assume_sorted=True,
            )
            self._interp = self._interp_s

    def forward(self, t_query: torch.Tensor) -> torch.Tensor | None:
        return self._interp(t_query)

    def _interp_none(self, t_query: torch.Tensor) -> None:
        return None

    def _interp_0(self, t_query: torch.Tensor) -> torch.Tensor:
        idx = int(torch.searchsorted(self.t, t_query).clamp(1, self.t.numel() - 1).item())
        return self._slice_u(idx - 1)

    def _interp_1(self, t_query: torch.Tensor) -> torch.Tensor:
        idx = int(torch.searchsorted(self.t, t_query).clamp(1, self.t.numel() - 1).item())
        t0, t1 = self.t[idx - 1], self.t[idx]
        u0, u1 = self._slice_u(idx - 1), self._slice_u(idx)
        w = (t_query - t0) / (t1 - t0)
        return (1.0 - w) * u0 + w * u1

    def _interp_s(self, t_query: torch.Tensor) -> torch.Tensor:
        uq = self._spl(t_query.detach().cpu().numpy())
        u = self.u
        if u is None:
            raise ValueError("Scipy control interpolation requires a control tensor.")
        return torch.as_tensor(uq, device=t_query.device, dtype=u.dtype)

    def _slice_u(self, idx: int) -> torch.Tensor:
        u = self.u
        if u is None:
            raise ValueError("Control interpolation requires a control tensor.")
        slices: list[slice | int] = [slice(None)] * u.ndim
        slices[self.axis] = idx
        return u[tuple(slices)]
