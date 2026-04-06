import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dymad.modules.kernel import KernelOpTangent


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

    def __init__(self, kernel, ridge_init=0, jitter=1e-10, device=None):
        super().__init__()
        self.kernel = kernel
        if isinstance(kernel, (nn.ModuleList, list)):
            self.dtype = kernel[0].dtype
        else:
            self.dtype = kernel.dtype
        self.device = device

        self.ridge_init = ridge_init
        self.jitter = float(jitter)

        # Train data & caches
        self.X_train = None
        self.Y_train = None
        self._residual = None

        # Placeholder for nn.Parameter
        # To be materialized in subclasses
        self.register_parameter(
            "_ridge_unconstrained",
            nn.Parameter(torch.empty(0, dtype=self.dtype, device=self.device)),
        )
        self.register_parameter(
            "_alphas", nn.Parameter(torch.empty(0, dtype=self.dtype, device=self.device))
        )
        self.register_parameter(
            "_Xref", nn.Parameter(torch.empty(0, dtype=self.dtype, device=self.device))
        )

    @property
    def ridge(self):
        return F.softplus(self._ridge_unconstrained)

    def set_train_data(self, X: torch.Tensor, Y: torch.Tensor):
        assert X.ndim == 2 and Y.ndim == 2
        self.X_train = torch.tensor(X, dtype=self.dtype, device=self.device)
        self.Y_train = torch.tensor(Y, dtype=self.dtype, device=self.device)
        self._Ndat, self._Dy = self.Y_train.shape
        self._residual = None  # reset

        if isinstance(self.kernel, list):
            for k in self.kernel:
                k.set_reference_data(self.X_train)
        else:
            self.kernel.set_reference_data(self.X_train)

        self._on_set_train_data()  # hook for subclasses

    def _on_set_train_data(self):
        pass  # optional in subclasses

    def fit(self):
        """
        Precompute the linear solve, which can be backprop'd.
        """
        return self._ensure_solved()

    def forward(self, Xnew: torch.Tensor):
        return self._predict_from_solution(Xnew)

    def _comp_residual(self):
        """
        Return the training residual after fit().
        """
        Ypred = self._predict_from_solution(self.X_train)
        return torch.linalg.norm(self.Y_train - Ypred) / np.sqrt(self._Ndat)

    def _ensure_solved(self):
        raise NotImplementedError("This is the base class.")

    def _predict_from_solution(self, Xnew: torch.Tensor):
        raise NotImplementedError("This is the base class.")


class KRRMultiOutputShared(KRRBase):
    """
    Scalar KRR for multiple outputs but one single kernel

        - One NxN Cholesky; solve `Dy` outputs together. One lambda (scalar) by default.
    """

    def __init__(self, kernel, ridge_init=0, jitter=1e-10, device=None):
        assert not kernel.is_operator_valued, "kernel should be scalar-valued."

        super().__init__(kernel, ridge_init=ridge_init, jitter=jitter, device=device)

        self._ridge_unconstrained = nn.Parameter(
            torch.tensor(ridge_init, dtype=self.dtype, device=self.device).log()
        )

    def __repr__(self) -> str:
        _s = self.kernel.__repr__()
        return (
            f"KRRMultiOutputShared(\n\tridge={self.ridge},\n\tjitter={self.jitter},\n\tdtype={self.dtype})"
            f"\n\twith:\n\t\tkernel={_s}"
        )

    def _on_set_train_data(self):
        self._alphas.requires_grad = False
        self._alphas.set_(torch.empty((self._Ndat, self._Dy), dtype=self.dtype, device=self.device))
        self._alphas.requires_grad = True
        self._Xref.requires_grad = False
        self._Xref.set_(self.X_train)
        self._Xref.requires_grad = False

    def _ensure_solved(self):
        if self._residual is not None:
            return self._residual

        assert self.X_train is not None and self.Y_train is not None, "Call set_train_data first."

        X, Y = self.X_train, self.Y_train
        Kxx = self.kernel(X, None)  # (N,N)
        I = torch.eye(self._Ndat, dtype=self.dtype, device=self.device)
        L = torch.linalg.cholesky(Kxx + (self.ridge + self.jitter) * I)
        A = torch.cholesky_solve(Y, L)  # (N,Dy)
        self._alphas.data.copy_(A)

        self._residual = self._comp_residual()
        return self._residual

    def _predict_from_solution(self, Xnew: torch.Tensor):
        Kxz = self.kernel(Xnew, self._Xref)  # (M,N)
        return Kxz @ self._alphas  # (M,Dy)


class KRRMultiOutputIndep(KRRBase):
    """
    Scalar KRR for multiple outputs, and one kernel per output

        - A ModuleList of `Dy` scalar kernels (one per output).
        - `Dy` independent NxN Choleskys; `Dy` ridges (vector).
    """

    def __init__(self, kernel, ridge_init=0, jitter=1e-10, device=None):
        assert isinstance(kernel, (nn.ModuleList, list)), "kernel should be a list of kernels."
        for _k in kernel:
            assert not _k.is_operator_valued, "kernel should be scalar-valued."

        super().__init__(kernel, ridge_init=ridge_init, jitter=jitter, device=device)

    def __repr__(self) -> str:
        _r = self.ridge_init if self.X_train is None else self.ridge
        _b = f", \n\tridge={_r},\n\tjitter={self.jitter},\n\tdtype={self.dtype})"
        _s = [k.__repr__() for k in self.kernel]
        return "KRRMultiOutputIndep(" + _b + "\n\twith:\n\t\t" + "\n\t\t".join(_s)

    def _on_set_train_data(self):
        self._alphas.requires_grad = False
        self._alphas.set_(torch.empty((self._Ndat, self._Dy), dtype=self.dtype, device=self.device))
        self._alphas.requires_grad = True
        self._Xref.requires_grad = False
        self._Xref.set_(self.X_train)
        self._Xref.requires_grad = False

        # per-output ridge vector (Dy,)
        if len(self._ridge_unconstrained) == 0:
            self._ridge_unconstrained.requires_grad = False
            if isinstance(self.ridge_init, (float, int)):
                self._ridge_unconstrained.set_(
                    torch.full(
                        (self._Dy,), self.ridge_init, dtype=self.dtype, device=self.device
                    ).log()
                )
            else:
                assert len(self.ridge_init) == self._Dy
                self._ridge_unconstrained.set_(
                    torch.tensor(self.ridge_init, dtype=self.dtype, device=self.device).log()
                )

    def _ensure_solved(self):
        if self._residual is not None:
            return self._residual

        assert self.X_train is not None and self.Y_train is not None, "Call set_train_data first."

        X, Y = self.X_train, self.Y_train
        assert isinstance(self.kernel, (nn.ModuleList, list)) and len(self.kernel) == self._Dy
        A = torch.empty_like(Y)
        I = torch.eye(self._Ndat, dtype=self.dtype, device=self.device)
        for d in range(self._Dy):
            Kxx = self.kernel[d](X, None)  # (N,N)
            L = torch.linalg.cholesky(Kxx + (self.ridge[d] + self.jitter) * I)
            A[:, d] = torch.cholesky_solve(Y[:, d : d + 1], L).squeeze(-1)
        self._alphas.data.copy_(A)

        self._residual = self._comp_residual()
        return self._residual

    def _predict_from_solution(self, Xnew: torch.Tensor):
        M = Xnew.shape[:-1]
        D = len(self.kernel)
        Yhat = torch.empty((*M, D), dtype=self.dtype, device=self.device)
        for d in range(D):
            Kxz = self.kernel[d](Xnew, self._Xref)
            Yhat[..., d] = Kxz @ self._alphas[:, d]
        return Yhat


class KRROperatorValued(KRRBase):
    """
    Operator-valued kernel K(X,Z) -> (N,M,Dy,Dy).

    Solves (Kxx + lambda I) vec(alpha) = vec(Y), using a single (N*Dy)x(N*Dy) Cholesky.
    """

    def __init__(self, kernel, ridge_init=0, jitter=1e-10, device=None):
        assert kernel.is_operator_valued, "kernel must be operator-valued."

        super().__init__(kernel, ridge_init=ridge_init, jitter=jitter, device=device)

        self._ridge_unconstrained = nn.Parameter(
            torch.tensor(ridge_init, dtype=self.dtype, device=self.device).log()
        )

    def __repr__(self) -> str:
        _s = self.kernel.__repr__()
        return f"KRROperatorValued(\n\tridge={self.ridge},\n\tjitter={self.jitter},\n\tdtype={self.dtype})\n\twith:\n\tkernel={_s}"

    def _on_set_train_data(self):
        self._alphas.requires_grad = False
        self._alphas.set_(torch.empty(self._Ndat * self._Dy, dtype=self.dtype, device=self.device))
        self._alphas.requires_grad = True
        self._Xref.requires_grad = False
        self._Xref.set_(self.X_train)
        self._Xref.requires_grad = False

    def _ensure_solved(self):
        if self._residual is not None:
            return self._residual

        assert self.X_train is not None and self.Y_train is not None, "Call set_train_data first."

        X, Y = self.X_train, self.Y_train

        Kxx = self.kernel(X, None)  # (N,N,Dy,Dy)
        Kflat = Kxx.reshape(self._Ndat * self._Dy, self._Ndat * self._Dy)
        A = Kflat + self.ridge * torch.eye(
            self._Ndat * self._Dy, dtype=self.dtype, device=self.device
        )
        L = torch.linalg.cholesky(
            A + self.jitter * torch.eye(A.size(0), dtype=self.dtype, device=self.device)
        )
        _tmp = torch.cholesky_solve(Y.reshape(-1, 1), L).squeeze(-1)  # (N*Dy,)
        self._alphas.data.copy_(_tmp)

        self._residual = self._comp_residual()
        return self._residual

    def _predict_from_solution(self, Xnew: torch.Tensor):
        dim = Xnew.shape[:-2]
        Kxz = self.kernel(Xnew, self._Xref)  # (...,M,Dy,N,Dy)
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

    def __init__(self, kernel, ridge_init=0, jitter=1e-10, device=None):
        assert isinstance(kernel, KernelOpTangent), "kernel must be KernelOpTangent."

        super().__init__(kernel, ridge_init=ridge_init, jitter=jitter, device=device)

        self._ridge_unconstrained = nn.Parameter(
            torch.tensor(ridge_init, dtype=self.dtype, device=self.device).log()
        )

    def __repr__(self) -> str:
        _s = self.kernel.__repr__()
        return f"KRRTangent(\n\tridge={self.ridge},\n\tjitter={self.jitter},\n\tdtype={self.dtype})\n\twith:\n\tkernel={_s}"

    def _on_set_train_data(self):
        self._Xref.requires_grad = False
        self._Xref.set_(self.X_train)
        self._Xref.requires_grad = False

    def set_manifold(self, manifold) -> None:
        # Only requires manifold to provide an estimate_tangent method
        # which can operate in batch, and give tangent bases of shape (...,d,Dy)
        self.kernel.set_manifold(manifold)

        self._alphas.requires_grad = False
        self._alphas.set_(
            torch.empty(self._Ndat * manifold._Nman, dtype=self.dtype, device=self.device)
        )
        self._alphas.requires_grad = True

    def _ensure_solved(self):
        if self._residual is not None:
            return self._residual

        assert self.X_train is not None and self.Y_train is not None, "Call set_train_data first."

        X, Y = self.X_train, self.Y_train

        Kxx, Tx, _ = self.kernel(X, None)  # (N,N,d,d), (N,d,Dy)
        _Y = torch.matmul(Tx, Y[..., None]).squeeze(-1)  # (N,d), the effective targets
        _d = _Y.shape[-1]

        Kflat = Kxx.reshape(self._Ndat * _d, self._Ndat * _d)
        A = Kflat + self.ridge * torch.eye(self._Ndat * _d, dtype=self.dtype, device=self.device)
        L = torch.linalg.cholesky(
            A + self.jitter * torch.eye(A.size(0), dtype=self.dtype, device=self.device)
        )
        _tmp = torch.cholesky_solve(_Y.reshape(-1, 1), L).squeeze(-1)  # (N*d,)
        self._alphas.data.copy_(_tmp)

        self._residual = self._comp_residual()
        return self._residual

    def _predict_from_solution(self, Xnew: torch.Tensor):
        dim = Xnew.shape[:-2]
        Kxz, Tx, _ = self.kernel(Xnew, self._Xref)  # (...,M,d,N,d), (...,M,d,Dy), (N,d,Dy)
        M, D, N, _ = Kxz.shape[-4:]
        Kflat = Kxz.reshape(*dim, M * D, N * D)  # (..., M*d, N*d)
        ynew_vec = Kflat @ self._alphas  # (..., M*d)
        _ynew = ynew_vec.reshape(*dim, M, D)  # (..., M, d)
        return torch.matmul(_ynew[..., None, :], Tx).squeeze(-2)  # (..., M, Dy)
