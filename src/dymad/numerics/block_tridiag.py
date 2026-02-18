from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, Tuple

import torch

logger = logging.getLogger(__name__)


def _raise_value_error(msg: str) -> None:
    logger.error(msg)
    raise ValueError(msg)


def _check_btd_shapes(D: torch.Tensor, U: torch.Tensor, L: torch.Tensor | None = None) -> Tuple[int, int]:
    if D.ndim < 3:
        _raise_value_error(f"D must have shape (..., T, n, n), got {tuple(D.shape)}.")
    T, n = D.shape[-3], D.shape[-1]
    if D.shape[-2] != n:
        _raise_value_error(f"D must have square blocks on the last two dimensions, got {tuple(D.shape)}.")
    if U.shape[-3] != T - 1 or U.shape[-2] != n or U.shape[-1] != n:
        _raise_value_error(
            f"U must have shape (..., T-1, n, n) consistent with D; got D={tuple(D.shape)}, U={tuple(U.shape)}."
        )
    if U.shape[:-3] != D.shape[:-3]:
        _raise_value_error(
            f"U batch dimensions must match D batch dimensions; got D={tuple(D.shape)}, U={tuple(U.shape)}."
        )
    if L is not None:
        if L.shape[-3] != T - 1 or L.shape[-2] != n or L.shape[-1] != n:
            _raise_value_error(
                f"L must have shape (..., T-1, n, n) consistent with D; got D={tuple(D.shape)}, L={tuple(L.shape)}."
            )
        if L.shape[:-3] != D.shape[:-3]:
            _raise_value_error(
                f"L batch dimensions must match D batch dimensions; got D={tuple(D.shape)}, L={tuple(L.shape)}."
            )
    return T, n


def _rhs_to_matrix(rhs: torch.Tensor, T: int, n: int) -> Tuple[torch.Tensor, bool]:
    if rhs.ndim >= 2 and rhs.shape[-2] == T and rhs.shape[-1] == n:
        return rhs.unsqueeze(-1), True
    if rhs.ndim >= 3 and rhs.shape[-3] == T and rhs.shape[-2] == n:
        return rhs, False
    _raise_value_error(
        f"rhs must have shape (..., T, n) or (..., T, n, m); got rhs={tuple(rhs.shape)}, T={T}, n={n}."
    )


@dataclass
class BTDLUFactors:
    D_tilde: torch.Tensor
    U: torch.Tensor
    M: torch.Tensor
    meta: Dict[str, Any] = field(default_factory=dict)


def factorize_btd_lu(D: torch.Tensor, U: torch.Tensor, L: torch.Tensor) -> BTDLUFactors:
    T, _ = _check_btd_shapes(D, U, L)
    D_tilde = D.clone()
    Uc = U.clone()
    M = torch.empty_like(L)

    for k in range(T - 1):
        Dk = D_tilde[..., k, :, :]
        Lk = L[..., k, :, :]
        Mk_t = torch.linalg.solve(Dk.transpose(-1, -2), Lk.transpose(-1, -2))
        Mk = Mk_t.transpose(-1, -2)
        M[..., k, :, :] = Mk
        D_tilde[..., k + 1, :, :] = D_tilde[..., k + 1, :, :] - Mk @ Uc[..., k, :, :]

    return BTDLUFactors(D_tilde=D_tilde, U=Uc, M=M)


def solve_btd_lu(factors: BTDLUFactors, b: torch.Tensor) -> torch.Tensor:
    D_tilde = factors.D_tilde
    U = factors.U
    M = factors.M
    T, n = D_tilde.shape[-3], D_tilde.shape[-1]
    rhs, is_vector = _rhs_to_matrix(b, T, n)
    if rhs.shape[:-3] != D_tilde.shape[:-3]:
        _raise_value_error(
            "rhs batch dimensions must match factor batch dimensions; "
            f"got D_tilde={tuple(D_tilde.shape)}, rhs={tuple(rhs.shape)}."
        )

    b_tilde = rhs.clone()
    for k in range(T - 1):
        b_tilde[..., k + 1, :, :] = b_tilde[..., k + 1, :, :] - M[..., k, :, :] @ b_tilde[..., k, :, :]

    x = torch.empty_like(rhs)
    x[..., T - 1, :, :] = torch.linalg.solve(D_tilde[..., T - 1, :, :], b_tilde[..., T - 1, :, :])
    for k in range(T - 2, -1, -1):
        rhs_k = b_tilde[..., k, :, :] - U[..., k, :, :] @ x[..., k + 1, :, :]
        x[..., k, :, :] = torch.linalg.solve(D_tilde[..., k, :, :], rhs_k)

    return x.squeeze(-1) if is_vector else x


@dataclass
class BTDSPDFactors:
    chol: torch.Tensor
    W: torch.Tensor
    meta: Dict[str, Any] = field(default_factory=dict)


def factorize_btd_spd(D: torch.Tensor, U: torch.Tensor) -> BTDSPDFactors:
    T, _ = _check_btd_shapes(D, U)
    chol = torch.empty_like(D)
    W = torch.empty_like(U)

    S0 = D[..., 0, :, :]
    chol0 = torch.linalg.cholesky(S0)
    chol[..., 0, :, :] = chol0
    if T > 1:
        W[..., 0, :, :] = torch.cholesky_solve(U[..., 0, :, :], chol0)

    for k in range(1, T):
        Sk = D[..., k, :, :] - U[..., k - 1, :, :].transpose(-1, -2) @ W[..., k - 1, :, :]
        chol_k = torch.linalg.cholesky(Sk)
        chol[..., k, :, :] = chol_k
        if k < T - 1:
            W[..., k, :, :] = torch.cholesky_solve(U[..., k, :, :], chol_k)

    return BTDSPDFactors(chol=chol, W=W)


def solve_btd_spd(factors: BTDSPDFactors, b: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    chol = factors.chol
    W = factors.W
    T, n = chol.shape[-3], chol.shape[-1]
    _check_btd_shapes(chol, U)
    rhs, is_vector = _rhs_to_matrix(b, T, n)
    if rhs.shape[:-3] != chol.shape[:-3]:
        _raise_value_error(
            "rhs batch dimensions must match factor batch dimensions; "
            f"got chol={tuple(chol.shape)}, rhs={tuple(rhs.shape)}."
        )

    y = torch.empty_like(rhs)
    y[..., 0, :, :] = torch.cholesky_solve(rhs[..., 0, :, :], chol[..., 0, :, :])
    for k in range(T - 1):
        rhs_next = rhs[..., k + 1, :, :] - U[..., k, :, :].transpose(-1, -2) @ y[..., k, :, :]
        y[..., k + 1, :, :] = torch.cholesky_solve(rhs_next, chol[..., k + 1, :, :])

    x = torch.empty_like(rhs)
    x[..., T - 1, :, :] = y[..., T - 1, :, :]
    for k in range(T - 2, -1, -1):
        x[..., k, :, :] = y[..., k, :, :] - W[..., k, :, :] @ x[..., k + 1, :, :]

    return x.squeeze(-1) if is_vector else x
