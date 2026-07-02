from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dymad.modules.helpers import _swap_parameter_storage
from dymad.modules.kernel import KernelAbstract, KernelOpTangent, inv_softplus
from dymad.numerics import conjugate_gradient_spd

KRRSolver = Literal["dense_cholesky", "matrix_free_cg", "auto"]


class KRRBase(nn.Module):
    """
    Base class for Kernel Ridge Regression, in particular:

        - Multi-output single scalar kernel (the most common case)
        - Multi-output multiple scalar kernel (i.e., one kernel per output)
        - True operator-valued kernel (i.e., matrix-valued)

    Subclasses must implement:

        - _ensure_solved(self)
        - _predict_from_solution(self, Xnew) -> (M, Dy)
    """

    def __init__(
        self,
        kernel: KernelAbstract | nn.ModuleList,
        ridge_init: float | list[float] = 0,
        jitter: float = 0.0,
        device: torch.device | str | None = None,
        solver: KRRSolver = "dense_cholesky",
        cg_rtol: float = 1.0e-10,
        cg_atol: float = 0.0,
        cg_max_iter: int = 1000,
        dense_threshold: int = 16_000_000,
    ):
        super().__init__()
        self.kernel: KernelAbstract | nn.ModuleList = kernel
        if isinstance(kernel, nn.ModuleList):
            self.dtype: torch.dtype = cast(torch.dtype, cast(Any, kernel[0]).dtype)
        else:
            self.dtype = cast(torch.dtype, kernel.dtype)
        self.device: torch.device | str | None = device

        self.ridge_init: float | list[float] = ridge_init
        self.jitter = float(jitter)
        if solver not in {"dense_cholesky", "matrix_free_cg", "auto"}:
            raise ValueError("solver must be 'dense_cholesky', 'matrix_free_cg', or 'auto'.")
        self.solver: KRRSolver = solver
        self.cg_rtol = float(cg_rtol)
        self.cg_atol = float(cg_atol)
        self.cg_max_iter = int(cg_max_iter)
        self.dense_threshold = int(dense_threshold)
        self._solver_used: str | None = None
        self._cg_diagnostics: dict[str, Any] | None = None

        # Train data & caches
        self.X_train: torch.Tensor | None = None
        self.Y_train: torch.Tensor | None = None
        self._residual: torch.Tensor | None = None

        # Placeholder for nn.Parameter
        # To be materialized in subclasses
        self._ridge_unconstrained: nn.Parameter = nn.Parameter(
            torch.empty(0, dtype=self.dtype, device=self.device)
        )
        self._alphas: nn.Parameter = nn.Parameter(
            torch.empty(0, dtype=self.dtype, device=self.device)
        )
        self._Xref: nn.Parameter = nn.Parameter(
            torch.empty(0, dtype=self.dtype, device=self.device), requires_grad=False
        )

    @property
    def ridge(self) -> torch.Tensor:
        return F.softplus(self._ridge_unconstrained)

    def set_train_data(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        assert X.ndim == 2 and Y.ndim == 2
        self.X_train = torch.as_tensor(X, dtype=self.dtype, device=self.device)
        self.Y_train = torch.as_tensor(Y, dtype=self.dtype, device=self.device)
        self._Ndat, self._Dy = self.Y_train.shape
        self._residual = None  # reset

        if isinstance(self.kernel, nn.ModuleList):
            for k in self.kernel:
                cast(KernelAbstract, k).set_reference_data(self.X_train)
        else:
            self.kernel.set_reference_data(self.X_train)

        self._on_set_train_data()  # hook for subclasses

    def set_reference_data(self, X: torch.Tensor) -> None:
        self.X_train = torch.as_tensor(X, dtype=self.dtype, device=self.device)
        self.Y_train = None
        self._residual = None

        if isinstance(self.kernel, nn.ModuleList):
            for k in self.kernel:
                cast(KernelAbstract, k).set_reference_data(self.X_train)
        else:
            self.kernel.set_reference_data(self.X_train)

        _swap_parameter_storage(self._Xref, self.X_train, requires_grad=False)
        _swap_parameter_storage(
            self._alphas,
            torch.empty(0, dtype=self.dtype, device=self.device),
            requires_grad=True,
        )

    def _on_set_train_data(self) -> None:
        pass  # optional in subclasses

    def fit(self) -> torch.Tensor:
        """
        Precompute the linear solve, which can be backprop'd.
        """
        return self._ensure_solved()

    def forward(self, Xnew: torch.Tensor):
        return self._predict_from_solution(Xnew)

    def _comp_residual(self) -> torch.Tensor:
        """
        Return the training residual after fit().
        """
        assert self.X_train is not None and self.Y_train is not None
        Ypred = self._predict_from_solution(self.X_train)
        return torch.linalg.norm(self.Y_train - Ypred) / np.sqrt(self._Ndat)

    def _ensure_solved(self) -> torch.Tensor:
        raise NotImplementedError("This is the base class.")

    def _predict_from_solution(self, Xnew: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("This is the base class.")

    def _require_reference_data(self) -> None:
        if self._Xref.numel() == 0:
            raise RuntimeError("Call set_reference_data or set_train_data before this operation.")

    def _use_dense_solver(self, materialized_elements: int) -> bool:
        if self.solver == "dense_cholesky":
            return True
        if self.solver == "matrix_free_cg":
            return False
        return materialized_elements <= self.dense_threshold

    def _solve_cg(self, matvec: Any, rhs: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        solution, diagnostics = conjugate_gradient_spd(
            matvec,
            rhs,
            rtol=self.cg_rtol,
            atol=self.cg_atol,
            max_iter=self.cg_max_iter,
        )
        if not diagnostics["converged"]:
            raise RuntimeError(
                "Matrix-free KRR CG solve did not converge "
                f"after {diagnostics['iterations']} iterations "
                f"(residual={diagnostics['residual_norm']:.3e}, "
                f"threshold={diagnostics['threshold']:.3e})."
            )
        self._cg_diagnostics = diagnostics
        return solution, diagnostics


class KRRMultiOutputShared(KRRBase):
    """
    Scalar KRR for multiple outputs but one single kernel

        - One NxN Cholesky; solve `Dy` outputs together. One lambda (scalar) by default.
    """

    def __init__(
        self,
        kernel: KernelAbstract,
        ridge_init: float = 0,
        jitter: float = 0.0,
        device: torch.device | str | None = None,
        solver: KRRSolver = "dense_cholesky",
        cg_rtol: float = 1.0e-10,
        cg_atol: float = 0.0,
        cg_max_iter: int = 1000,
        dense_threshold: int = 16_000_000,
    ):
        assert not kernel.is_operator_valued, "kernel should be scalar-valued."

        super().__init__(
            kernel,
            ridge_init=ridge_init,
            jitter=jitter,
            device=device,
            solver=solver,
            cg_rtol=cg_rtol,
            cg_atol=cg_atol,
            cg_max_iter=cg_max_iter,
            dense_threshold=dense_threshold,
        )

        self._ridge_unconstrained = nn.Parameter(
            inv_softplus(ridge_init, self.dtype, device=self.device)
        )

    def __repr__(self) -> str:
        _s = self.kernel.__repr__()
        return (
            f"KRRMultiOutputShared(\n\tridge={self.ridge},\n\tjitter={self.jitter},\n\tdtype={self.dtype})"
            f"\n\twith:\n\t\tkernel={_s}"
        )

    def _on_set_train_data(self) -> None:
        assert self.X_train is not None
        _swap_parameter_storage(
            self._alphas,
            torch.empty((self._Ndat, self._Dy), dtype=self.dtype, device=self.device),
            requires_grad=True,
        )
        _swap_parameter_storage(self._Xref, self.X_train, requires_grad=False)

    def _ensure_solved(self) -> torch.Tensor:
        if self._residual is not None:
            return self._residual

        assert self.X_train is not None and self.Y_train is not None, "Call set_train_data first."

        X, Y = self.X_train, self.Y_train
        if not self._use_dense_solver(self._Ndat * self._Ndat):
            ridge = self.ridge + self.jitter

            def matvec(values: torch.Tensor) -> torch.Tensor:
                return cast(KernelAbstract, self.kernel).apply(X, X, values) + ridge * values

            A, _diagnostics = self._solve_cg(matvec, Y)
            self._alphas.data.copy_(A)
            self._solver_used = "matrix_free_cg"
            self._residual = self._comp_residual()
            return self._residual

        Kxx = cast(KernelAbstract, self.kernel)(X, None)  # (N,N)
        I = torch.eye(self._Ndat, dtype=self.dtype, device=self.device)
        L = torch.linalg.cholesky(Kxx + (self.ridge + self.jitter) * I)
        A = torch.cholesky_solve(Y, L)  # (N,Dy)
        self._alphas.data.copy_(A)

        self._solver_used = "dense_cholesky"
        self._residual = self._comp_residual()
        return self._residual

    def _predict_from_solution(self, Xnew: torch.Tensor):
        if self._solver_used == "matrix_free_cg":
            return cast(KernelAbstract, self.kernel).apply(Xnew, self._Xref, self._alphas)
        Kxz = cast(KernelAbstract, self.kernel)(Xnew, self._Xref)  # (M,N)
        return Kxz @ self._alphas  # (M,Dy)


class KRRMultiOutputIndep(KRRBase):
    """
    Scalar KRR for multiple outputs, and one kernel per output

        - A ModuleList of `Dy` scalar kernels (one per output).
        - `Dy` independent NxN Choleskys; `Dy` ridges (vector).
    """

    def __init__(
        self,
        kernel: nn.ModuleList,
        ridge_init: float | list[float] = 0,
        jitter: float = 0.0,
        device: torch.device | str | None = None,
        solver: KRRSolver = "dense_cholesky",
        cg_rtol: float = 1.0e-10,
        cg_atol: float = 0.0,
        cg_max_iter: int = 1000,
        dense_threshold: int = 16_000_000,
    ):
        assert isinstance(kernel, nn.ModuleList), "kernel should be a ModuleList of kernels."
        for _k in kernel:
            assert not cast(KernelAbstract, _k).is_operator_valued, (
                "kernel should be scalar-valued."
            )

        super().__init__(
            kernel,
            ridge_init=ridge_init,
            jitter=jitter,
            device=device,
            solver=solver,
            cg_rtol=cg_rtol,
            cg_atol=cg_atol,
            cg_max_iter=cg_max_iter,
            dense_threshold=dense_threshold,
        )

    def __repr__(self) -> str:
        _r = self.ridge_init if self.X_train is None else self.ridge
        _b = f", \n\tridge={_r},\n\tjitter={self.jitter},\n\tdtype={self.dtype})"
        assert isinstance(self.kernel, nn.ModuleList)
        _s = [k.__repr__() for k in self.kernel]
        return "KRRMultiOutputIndep(" + _b + "\n\twith:\n\t\t" + "\n\t\t".join(_s)

    def _on_set_train_data(self) -> None:
        assert self.X_train is not None
        _swap_parameter_storage(
            self._alphas,
            torch.empty((self._Ndat, self._Dy), dtype=self.dtype, device=self.device),
            requires_grad=True,
        )
        _swap_parameter_storage(self._Xref, self.X_train, requires_grad=False)

        # per-output ridge vector (Dy,)
        if len(self._ridge_unconstrained) == 0:
            if isinstance(self.ridge_init, (float, int)):
                ridge_unconstrained = torch.full(
                    (self._Dy,), float(self.ridge_init), dtype=self.dtype, device=self.device
                )
            else:
                assert len(self.ridge_init) == self._Dy
                ridge_unconstrained = torch.tensor(
                    self.ridge_init, dtype=self.dtype, device=self.device
                )
            ridge_unconstrained = inv_softplus(ridge_unconstrained, self.dtype, device=self.device)
            _swap_parameter_storage(
                self._ridge_unconstrained, ridge_unconstrained, requires_grad=True
            )

    def _ensure_solved(self):
        if self._residual is not None:
            return self._residual

        assert self.X_train is not None and self.Y_train is not None, "Call set_train_data first."

        X, Y = self.X_train, self.Y_train
        assert isinstance(self.kernel, nn.ModuleList) and len(self.kernel) == self._Dy
        kernels = cast(nn.ModuleList, self.kernel)
        A = torch.empty_like(Y)
        if not self._use_dense_solver(self._Dy * self._Ndat * self._Ndat):
            for d in range(self._Dy):
                ridge = self.ridge[d] + self.jitter

                def matvec(values: torch.Tensor, d: int = d) -> torch.Tensor:
                    return cast(KernelAbstract, kernels[d]).apply(X, X, values) + ridge * values

                solution, _diagnostics = self._solve_cg(matvec, Y[:, d : d + 1])
                A[:, d] = solution.squeeze(-1)
            self._alphas.data.copy_(A)
            self._solver_used = "matrix_free_cg"
            self._residual = self._comp_residual()
            return self._residual

        I = torch.eye(self._Ndat, dtype=self.dtype, device=self.device)
        for d in range(self._Dy):
            Kxx = cast(KernelAbstract, kernels[d])(X, None)  # (N,N)
            L = torch.linalg.cholesky(Kxx + (self.ridge[d] + self.jitter) * I)
            A[:, d] = torch.cholesky_solve(Y[:, d : d + 1], L).squeeze(-1)
        self._alphas.data.copy_(A)

        self._solver_used = "dense_cholesky"
        self._residual = self._comp_residual()
        return self._residual

    def _predict_from_solution(self, Xnew: torch.Tensor):
        M = Xnew.shape[:-1]
        assert isinstance(self.kernel, nn.ModuleList)
        kernels = cast(nn.ModuleList, self.kernel)
        D = len(kernels)
        Yhat = torch.empty((*M, D), dtype=self.dtype, device=self.device)
        for d in range(D):
            if self._solver_used == "matrix_free_cg":
                Yhat[..., d] = (
                    cast(KernelAbstract, kernels[d])
                    .apply(Xnew, self._Xref, self._alphas[:, d : d + 1])
                    .squeeze(-1)
                )
            else:
                Kxz = cast(KernelAbstract, kernels[d])(Xnew, self._Xref)
                Yhat[..., d] = Kxz @ self._alphas[:, d]
        return Yhat


class KRROperatorValued(KRRBase):
    """
    Operator-valued kernel K(X,Z) -> (N,M,Dy,Dy).

    Solves (Kxx + lambda I) vec(alpha) = vec(Y), using a single (N*Dy)x(N*Dy) Cholesky.
    """

    def __init__(
        self,
        kernel: KernelAbstract,
        ridge_init: float = 0,
        jitter: float = 0.0,
        device: torch.device | str | None = None,
        solver: KRRSolver = "dense_cholesky",
        cg_rtol: float = 1.0e-10,
        cg_atol: float = 0.0,
        cg_max_iter: int = 1000,
        dense_threshold: int = 16_000_000,
    ):
        assert kernel.is_operator_valued, "kernel must be operator-valued."

        super().__init__(
            kernel,
            ridge_init=ridge_init,
            jitter=jitter,
            device=device,
            solver=solver,
            cg_rtol=cg_rtol,
            cg_atol=cg_atol,
            cg_max_iter=cg_max_iter,
            dense_threshold=dense_threshold,
        )

        self._ridge_unconstrained = nn.Parameter(
            inv_softplus(ridge_init, self.dtype, device=self.device)
        )

    def __repr__(self) -> str:
        _s = self.kernel.__repr__()
        return f"KRROperatorValued(\n\tridge={self.ridge},\n\tjitter={self.jitter},\n\tdtype={self.dtype})\n\twith:\n\tkernel={_s}"

    def _on_set_train_data(self) -> None:
        assert self.X_train is not None
        _swap_parameter_storage(
            self._alphas,
            torch.empty(self._Ndat * self._Dy, dtype=self.dtype, device=self.device),
            requires_grad=True,
        )
        _swap_parameter_storage(self._Xref, self.X_train, requires_grad=False)

    def _ensure_solved(self):
        if self._residual is not None:
            return self._residual

        assert self.X_train is not None and self.Y_train is not None, "Call set_train_data first."

        X, Y = self.X_train, self.Y_train

        if not self._use_dense_solver((self._Ndat * self._Dy) ** 2):
            ridge = self.ridge + self.jitter

            def matvec(flat_values: torch.Tensor) -> torch.Tensor:
                values = flat_values.reshape(self._Ndat, self._Dy)
                applied = cast(KernelAbstract, self.kernel).apply(X, X, values)
                return (applied + ridge * values).reshape(-1, 1)

            _tmp, _diagnostics = self._solve_cg(matvec, Y.reshape(-1, 1))
            self._alphas.data.copy_(_tmp.squeeze(-1))
            self._solver_used = "matrix_free_cg"
            self._residual = self._comp_residual()
            return self._residual

        Kxx = cast(KernelAbstract, self.kernel)(X, None)  # (N,N,Dy,Dy)
        Kflat = Kxx.reshape(self._Ndat * self._Dy, self._Ndat * self._Dy)
        A = Kflat + self.ridge * torch.eye(
            self._Ndat * self._Dy, dtype=self.dtype, device=self.device
        )
        L = torch.linalg.cholesky(
            A + self.jitter * torch.eye(A.size(0), dtype=self.dtype, device=self.device)
        )
        _tmp = torch.cholesky_solve(Y.reshape(-1, 1), L).squeeze(-1)  # (N*Dy,)
        self._alphas.data.copy_(_tmp)

        self._solver_used = "dense_cholesky"
        self._residual = self._comp_residual()
        return self._residual

    def _predict_from_solution(self, Xnew: torch.Tensor):
        if self._solver_used == "matrix_free_cg":
            return cast(KernelAbstract, self.kernel).apply(
                Xnew, self._Xref, self._alphas.reshape(self._Ndat, self._Dy)
            )
        dim = Xnew.shape[:-2]
        Kxz = cast(KernelAbstract, self.kernel)(Xnew, self._Xref)  # (...,M,Dy,N,Dy)
        M, D, N, _ = Kxz.shape[-4:]
        Kflat = Kxz.reshape(*dim, M * D, N * D)  # (..., M*Dy, N*Dy)
        ynew_vec = Kflat @ self._alphas  # (..., M*Dy)
        return ynew_vec.reshape(*dim, M, D)  # (..., M, Dy)


class KRRTangent(KRRBase):
    """
    KRR for vector fields on a manifold, using a specialized tangent kernel.

    The formulation is based on Geometrically constraint Multivariate KRR (GMKRR) from

        Huang, He, Harlim, Li, ICLR2025

    Solves (Kxx + lambda I) vec(alpha) = vec(Y), but Kxx is given in a factorized form,
    so effectively we solve a smaller system in intrinsic dimension d << Dy.

        (kxx + lambda I) vec(alpha) = vec(T^T * Y)
    """

    def __init__(
        self,
        kernel: KernelOpTangent,
        ridge_init: float = 0,
        jitter: float = 0.0,
        device: torch.device | str | None = None,
        solver: KRRSolver = "dense_cholesky",
        cg_rtol: float = 1.0e-10,
        cg_atol: float = 0.0,
        cg_max_iter: int = 1000,
        dense_threshold: int = 16_000_000,
    ):
        assert isinstance(kernel, KernelOpTangent), "kernel must be KernelOpTangent."

        super().__init__(
            kernel,
            ridge_init=ridge_init,
            jitter=jitter,
            device=device,
            solver=solver,
            cg_rtol=cg_rtol,
            cg_atol=cg_atol,
            cg_max_iter=cg_max_iter,
            dense_threshold=dense_threshold,
        )

        self._ridge_unconstrained = nn.Parameter(
            inv_softplus(ridge_init, self.dtype, device=self.device)
        )

    def __repr__(self) -> str:
        _s = self.kernel.__repr__()
        return f"KRRTangent(\n\tridge={self.ridge},\n\tjitter={self.jitter},\n\tdtype={self.dtype})\n\twith:\n\tkernel={_s}"

    def _on_set_train_data(self) -> None:
        assert self.X_train is not None
        _swap_parameter_storage(self._Xref, self.X_train, requires_grad=False)

    def set_manifold(self, manifold) -> None:
        # Only requires manifold to provide an estimate_tangent method
        # which can operate in batch, and give tangent bases of shape (...,d,Dy)
        cast(KernelOpTangent, self.kernel).set_manifold(manifold)

        _swap_parameter_storage(
            self._alphas,
            torch.empty(self._Ndat * manifold._Nman, dtype=self.dtype, device=self.device),
            requires_grad=True,
        )

    def _ensure_solved(self):
        if self._residual is not None:
            return self._residual

        assert self.X_train is not None and self.Y_train is not None, "Call set_train_data first."

        X, Y = self.X_train, self.Y_train

        if not self._use_dense_solver(self._Ndat * self._Ndat * self._Dy * self._Dy):
            Tx = cast(KernelOpTangent, self.kernel)._tangent(X)
            _Y = torch.matmul(Tx, Y[..., None]).squeeze(-1)
            _d = _Y.shape[-1]
            ridge = self.ridge + self.jitter

            def matvec(flat_values: torch.Tensor) -> torch.Tensor:
                values = flat_values.reshape(self._Ndat, _d)
                applied, _ = cast(KernelOpTangent, self.kernel).intrinsic_apply(
                    X, X, values, Tx=Tx, Tz=Tx
                )
                return (applied + ridge * values).reshape(-1, 1)

            _tmp, _diagnostics = self._solve_cg(matvec, _Y.reshape(-1, 1))
            self._alphas.data.copy_(_tmp.squeeze(-1))
            self._solver_used = "matrix_free_cg"
            self._residual = self._comp_residual()
            return self._residual

        Kxx, Tx, _ = cast(KernelOpTangent, self.kernel)(X, None)  # (N,N,d,d), (N,d,Dy)
        _Y = torch.matmul(Tx, Y[..., None]).squeeze(-1)  # (N,d), the effective targets
        _d = _Y.shape[-1]

        Kflat = Kxx.reshape(self._Ndat * _d, self._Ndat * _d)
        A = Kflat + self.ridge * torch.eye(self._Ndat * _d, dtype=self.dtype, device=self.device)
        L = torch.linalg.cholesky(
            A + self.jitter * torch.eye(A.size(0), dtype=self.dtype, device=self.device)
        )
        _tmp = torch.cholesky_solve(_Y.reshape(-1, 1), L).squeeze(-1)  # (N*d,)
        self._alphas.data.copy_(_tmp)

        self._solver_used = "dense_cholesky"
        self._residual = self._comp_residual()
        return self._residual

    def _predict_from_solution(self, Xnew: torch.Tensor):
        if self._solver_used == "matrix_free_cg":
            Tx = cast(KernelOpTangent, self.kernel)._tangent(Xnew)
            Tz = cast(KernelOpTangent, self.kernel)._tangent(self._Xref)
            _d = Tz.shape[-2]
            intrinsic, Tx = cast(KernelOpTangent, self.kernel).intrinsic_apply(
                Xnew, self._Xref, self._alphas.reshape(self._Ndat, _d), Tx=Tx, Tz=Tz
            )
            return torch.matmul(intrinsic[..., None, :], Tx).squeeze(-2)
        dim = Xnew.shape[:-2]
        Kxz, Tx, _ = cast(KernelOpTangent, self.kernel)(
            Xnew, self._Xref
        )  # (...,M,d,N,d), (...,M,d,Dy), (N,d,Dy)
        M, D, N, _ = Kxz.shape[-4:]
        Kflat = Kxz.reshape(*dim, M * D, N * D)  # (..., M*d, N*d)
        ynew_vec = Kflat @ self._alphas  # (..., M*d)
        _ynew = ynew_vec.reshape(*dim, M, D)  # (..., M, d)
        return torch.matmul(_ynew[..., None, :], Tx).squeeze(-2)  # (..., M, Dy)
