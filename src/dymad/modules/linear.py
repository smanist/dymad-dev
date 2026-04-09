from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlexLinear(nn.Module):
    """
    A linear layer that can store weights either as a full matrix (MxN)
    or as low-rank factors (U, V) with efficient matvec operations.

    In the low-rank mode, the weight matrix is represented as:

        W = U @ V^T

    where U is (M x r) and V is (N x r).
    """

    def __init__(self, in_features, out_features, bias=True, dtype=None, device=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Full weight params
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=dtype, device=device)
        )
        self.bias = (
            nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device)) if bias else None
        )

        # Low-rank params
        self.U = nn.Parameter(torch.empty(0, 0, dtype=dtype, device=device), requires_grad=False)
        self.V = nn.Parameter(torch.empty(0, 0, dtype=dtype, device=device), requires_grad=False)

        self.mode = "full"
        self.rank = None

    def __repr__(self) -> str:
        return (
            f"FlexLinear(in_features={self.in_features}, out_features={self.out_features}, "
            + f"mode={self.mode}, rank={self.rank})"
        )

    def _init_linear(
        self,
        weight_init: str | Callable[..., Any] = nn.init.xavier_uniform_,
        bias_init: Callable[..., Any] = nn.init.zeros_,
        gain: float = 1.0,
    ) -> None:
        if isinstance(weight_init, str) or isinstance(bias_init, str):
            raise TypeError("Resolved init functions must be callables before FlexLinear init.")
        if self.mode == "full":
            weight_init(self.weight, gain)
        else:
            weight_init(self.U, gain)
            weight_init(self.V, gain)
        if self.bias is not None:
            bias_init(self.bias)

    @torch.no_grad()
    def set_full(self, W: torch.Tensor, b: torch.Tensor | None):
        """Switch to full mode and copy parameters."""
        assert W.shape == (self.out_features, self.in_features)
        self.mode = "full"
        self.rank = None
        # release low-rank
        self.U.requires_grad_(False)
        self.V.requires_grad_(False)
        self.U = nn.Parameter(
            torch.empty(0, 0, dtype=W.dtype, device=W.device), requires_grad=False
        )
        self.V = nn.Parameter(
            torch.empty(0, 0, dtype=W.dtype, device=W.device), requires_grad=False
        )
        # set full
        self.weight.data.copy_(W)
        if self.bias is not None and b is not None:
            self.bias.data.copy_(b)

    @torch.no_grad()
    def set_lora(self, U: torch.Tensor, V: torch.Tensor, b: torch.Tensor | None):
        """Switch to lowrank mode and copy factors. U: out*r, V: in*r."""
        assert U.shape[0] == self.out_features and V.shape[0] == self.in_features
        assert U.shape[1] == V.shape[1]
        self.rank = U.shape[1]
        self.mode = "lora"

        # freeze full weight (not used in forward)
        self.weight.requires_grad_(False)

        # (re)allocate factors with grad
        self.U = nn.Parameter(U.clone(), requires_grad=True)
        self.V = nn.Parameter(V.clone(), requires_grad=True)

        if self.bias is not None and b is not None:
            self.bias.data.copy_(b)

    @torch.no_grad()
    def set_weights(
        self,
        W: torch.Tensor | None = None,
        b: torch.Tensor | None = None,
        U: torch.Tensor | None = None,
        V: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        if W is not None:
            self.set_full(W, b)
            return (self.weight,) if self.bias is None else (self.weight, self.bias)
        elif U is not None and V is not None:
            self.set_lora(U, V, b)
            if self.bias is None:
                return self.U, self.V
            return self.U, self.V, self.bias
        else:
            raise ValueError("Must provide either full weights (W) or low-rank factors (U, V).")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "full":
            return F.linear(x, self.weight, self.bias)
        else:
            # Efficient matvec-only: (x @ V) @ U^T + b
            y = (x @ self.V) @ self.U.T
            if self.bias is not None:
                y = y + self.bias
            return y

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Depending on the running environment (e.g., GitHub CI), the keys may be prefixed differently.

        # Check if U and V exists - they should always appear in pair.
        if prefix + "U" in state_dict:
            U_ckpt = state_dict[prefix + "U"]
            V_ckpt = state_dict[prefix + "V"]
        elif prefix + "net.0.U" in state_dict:
            U_ckpt = state_dict[prefix + "net.0.U"]
            V_ckpt = state_dict[prefix + "net.0.V"]
        else:
            U_ckpt, V_ckpt = None, None

        # Check if weights exist.
        if prefix + "weight" in state_dict:
            W_ckpt = state_dict[prefix + "weight"]
        elif prefix + "net.0.weight" in state_dict:
            W_ckpt = state_dict[prefix + "net.0.weight"]
        else:
            W_ckpt = None

        # bias should always exist, but might prefixed differently.
        if prefix + "bias" in state_dict:
            b_ckpt = state_dict[prefix + "bias"]
        elif prefix + "net.0.bias" in state_dict:
            b_ckpt = state_dict[prefix + "net.0.bias"]
        else:
            b_ckpt = None

        # U and V, if exist, are empty if full mode, otherwise they are low-rank factors.
        if U_ckpt is None or V_ckpt is None:
            is_lowrank = False
        else:
            is_lowrank = U_ckpt.shape[0] > 0 and V_ckpt.shape[0] > 0

        # Set the parameters
        if is_lowrank:
            assert U_ckpt is not None and V_ckpt is not None
            if self.U.shape != U_ckpt.shape or self.V.shape != V_ckpt.shape:
                # re-register parameters with correct shapes
                device = self.weight.device
                self.U = nn.Parameter(
                    torch.empty_like(U_ckpt, dtype=U_ckpt.dtype, device=device), requires_grad=True
                )
                self.V = nn.Parameter(
                    torch.empty_like(V_ckpt, dtype=V_ckpt.dtype, device=device), requires_grad=True
                )
            self.U.data.copy_(U_ckpt)
            self.V.data.copy_(V_ckpt)
        else:
            assert W_ckpt is not None
            if self.weight.shape != W_ckpt.shape:
                # re-register parameters with correct shapes
                device = self.weight.device
                self.weight = nn.Parameter(
                    torch.empty_like(W_ckpt, dtype=W_ckpt.dtype, device=device)
                )
            self.weight.data.copy_(W_ckpt)

        if b_ckpt is not None:
            if self.bias is None:
                # re-register bias parameter
                device = self.weight.device
                self.bias = nn.Parameter(
                    torch.empty_like(b_ckpt, dtype=b_ckpt.dtype, device=device)
                )
            elif self.bias.shape != b_ckpt.shape:
                # re-register bias parameter with correct shape
                device = self.bias.device
                self.bias = nn.Parameter(
                    torch.empty_like(b_ckpt, dtype=b_ckpt.dtype, device=device)
                )
            self.bias.data.copy_(b_ckpt)

        self.mode = "lora" if is_lowrank else "full"
        self.rank = self.U.shape[1] if is_lowrank else None
