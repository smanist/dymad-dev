import logging
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import scipy.linalg as spl
import torch

from dymad.numerics.complex import disc2cont

logger = logging.getLogger(__name__)


def conjugate_gradient_spd(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    *,
    rtol: float = 1.0e-10,
    atol: float = 0.0,
    max_iter: int = 1000,
    initial: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Solve an SPD linear system with conjugate gradients."""
    if rtol < 0.0:
        raise ValueError("rtol must be non-negative.")
    if atol < 0.0:
        raise ValueError("atol must be non-negative.")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")

    x = torch.zeros_like(rhs) if initial is None else initial.clone()
    r = rhs - matvec(x)
    p = r.clone()
    rs_old = torch.sum(r * r)
    rhs_norm = torch.linalg.norm(rhs)
    threshold = torch.maximum(
        torch.as_tensor(atol, dtype=rhs.dtype, device=rhs.device),
        torch.as_tensor(rtol, dtype=rhs.dtype, device=rhs.device) * rhs_norm,
    )
    residual_norm = torch.sqrt(rs_old)
    if float(residual_norm.detach().cpu()) <= float(threshold.detach().cpu()):
        return x, {
            "converged": True,
            "iterations": 0,
            "residual_norm": float(residual_norm.detach().cpu()),
            "threshold": float(threshold.detach().cpu()),
        }

    converged = False
    iterations = 0
    tiny = torch.finfo(rhs.dtype).tiny
    for iteration in range(1, max_iter + 1):
        Ap = matvec(p)
        denom = torch.clamp(torch.sum(p * Ap), min=tiny)
        alpha = rs_old / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.sum(r * r)
        residual_norm = torch.sqrt(rs_new)
        iterations = iteration
        if float(residual_norm.detach().cpu()) <= float(threshold.detach().cpu()):
            converged = True
            break
        beta = rs_new / torch.clamp(rs_old, min=tiny)
        p = r + beta * p
        rs_old = rs_new

    return x, {
        "converged": converged,
        "iterations": iterations,
        "residual_norm": float(residual_norm.detach().cpu()),
        "threshold": float(threshold.detach().cpu()),
    }


def truncated_svd(X, order):
    """
    A vanilla interface for different types of truncation order.

    Possible order parameters

    - Float, positive: Energy percentage
    - Float, negative: Optimal truncation by Gavish&Donoho.
    - Integer, positive: Keep first N pairs
    - Integer, negative: Remove last N pairs
    - 'full': Retain all pairs
    """
    _U, _S, _Vh = np.linalg.svd(X, full_matrices=False)
    if isinstance(order, float):
        if order > 0:
            _s2 = _S**2
            _I = np.argmax(np.cumsum(_s2) / np.sum(_s2) > order)
        else:
            _n, _m = X.shape
            _bt = min(_n, _m) / max(_n, _m)
            _om = 0.56 * _bt**3 - 0.95 * _bt**2 + 1.82 * _bt + 1.43
            _I = np.argmax(_S < _om * np.median(_S))
        _Ur = _U[:, :_I]
        _Sr = _S[:_I]
        _Vr = _Vh[:_I].conj().T
    elif isinstance(order, int):
        _Ur = _U[:, :order]
        _Sr = _S[:order]
        _Vr = _Vh[:order].conj().T
    elif order.lower() == "full":
        _Ur, _Sr, _Vr = _U, _S, _Vh.conj().T
    else:
        raise NotImplementedError(f"Undefined threshold for order={order}")
    return _Ur, _Sr, _Vr


def randomized_svd(loader, N, k, oversample=10, n_iter=0, return_u=False, dtype=np.float64, seed=0):
    """
    Two-pass randomized SVD for a matrix stored as row-blocks:

    A^T = [A_0^T, A_1^T, ..., A_{N-1}^T]

    where A_i shape (n_i, d), A shape (N=sum_i n_i, d)

    Two passes estimate k singular values and right singular vectors.
    A third pass is needed if left singular vectors are requested.

    Args:
        loader (callable): A function that takes a block index and returns the corresponding block.
        N (int): Number of blocks
        k (int): Target rank (k > 0).
        oversample (int, default 10): Extra dimensions to improve spectral accuracy (total sketch dim l = k + oversample).
        n_iter (int, default 0): Number of power iterations (each adds two more passes).
        return_u (bool, default False): Whether to return left singular vectors.
        dtype (numpy dtype, default np.float64): Internal precision for computation.
        seed (int or None, default 0): RNG seed for the Gaussian sketch.

    Returns:
        U : (N, k) ndarray, Left singular vectors (columns).
        S : (k,) ndarray, Singular values in descending order.
        Vt : (d, k) ndarray, Right singular vectors (columns).
    """
    if k <= 0:
        raise ValueError("k must be positive")
    logger.info("Row-block randomized SVD starts...")

    # Determine feature dimension d
    d = loader(0).shape[1]
    l = k + oversample  # total sketch dimension

    rng = np.random.default_rng(seed)
    Omega = rng.standard_normal((d, l)).astype(dtype)

    # First pass: Y = Σ_i X_i^T (X_i @ Omega)
    logger.info("First pass:")
    Y = np.zeros((d, l), dtype=dtype)
    for _i in range(N):
        X = loader(_i).astype(dtype)
        Y += X.T.dot(X.dot(Omega))

    # Optional power iterations
    logger.info(f"{n_iter} power iteration(s):")
    for _ in range(n_iter):
        Z = np.zeros_like(Y)
        for _i in range(N):
            X = loader(_i).astype(dtype)
            Z += X.T.dot(X.dot(Y))
        Y = Z

    # Orthonormal basis Q for range(Y)
    Q, _ = np.linalg.qr(Y)

    # Second pass: B = (X Q)^T (X Q)
    logger.info("Second pass:")
    B = np.zeros((l, l), dtype=dtype)
    for _i in range(N):
        X = loader(_i).astype(dtype)
        T = X.dot(Q)
        B += T.T.dot(T)

    # Eigendecomposition of small matrix B
    logger.info("Eigendecomposition:")
    S2, W = np.linalg.eigh(B)  # ascending order
    idx = np.argsort(S2)[::-1][:k]
    S = np.sqrt(S2[idx]).astype(dtype)
    V = Q.dot(W[:, idx])

    # Optionally compute left singular vectors U
    if return_u:
        logger.info("Computing left singular vectors:")
        S_inv = np.zeros((k,), dtype=dtype)
        msk = S > 1e-15 * S[0]
        S_inv[msk] = 1.0 / S[msk]
        U = []
        for _i in range(N):
            X = loader(_i).astype(dtype)
            U.append(X.dot(V).dot(np.diag(S_inv)))
        U = np.vstack(U)

    logger.info("Done")

    if return_u:
        return U, S, V
    return S, V


def truncated_lstsq(A, B, tsvd=None):
    """
    Solve the linear system AX = B by least squares.

    If truncated SVD is used, the function returns the two factors of X.

    Args:
        A (np.ndarray): Coefficient matrix.
        B (np.ndarray): Right-hand side matrix.
        tsvd (int or float, optional): If provided, use truncated SVD with this order.

    Returns:
        np.ndarray: Solution matrix X, or its two factors.
    """
    if tsvd is None:
        return np.linalg.lstsq(A, B, rcond=None)[0]

    _Ur, _Sr, _Vr = truncated_svd(A, tsvd)
    _B = (_Ur.conj().T @ B) / _Sr.reshape(-1, 1)
    return _Vr, _B.T


def check_direction(v1, v2):
    """
    The cosine values between v1 and v2.
    """
    if len(v1.shape) == 1:
        # Just one vector
        return _check_direction(v1, v2)
    _, _N = v1.shape
    _d = np.zeros(
        _N,
    )
    for _i in range(_N):
        _d[_i] = _check_direction(v1[:, _i], v2[:, _i])
    return _d


def _check_direction(v1, v2):
    """
    Expecting the angle between v1 and v2 is zero, i.e., aligned.
    """
    _v1 = v1.reshape(-1)
    _v2 = v2.reshape(-1)
    _cc = _v1.conj().dot(_v2) / (np.linalg.norm(_v1) * np.linalg.norm(_v2))
    return np.abs(_cc)


def check_orthogonality(U, V, M=None):
    """
    Expecting U.H * M * V = I
    """
    _n, _m = U.shape
    if M is None:
        _M = np.eye(_n)
    else:
        _M = np.array(M)
    _L = U.conj().T.dot(_M).dot(V)
    _err = np.mean(np.abs(_L - np.eye(_m)))
    return _L, _err


def scaled_eig(A, B=None):
    """
    Suppose
    A U = B U L, V^H A = L V^H B
    Ideally one should have double diagonalization (for non-degenerate case):
    V^H B U = I and V^H A U = L
    but by default each column of U and V is normalized by the length, and the
    double diagonalization is not satisfied.
    Here we scale both U and V so that they are approximately orthonormal to each other
    (w.r.t. B); also the scaling is such that the norms of u_i and v_i are equal.

    However, if one needs to project quantities to, e.g., U, use pseudo-inverse of U
    instead of V for numerical robustness.
    """
    _eigvals, _left, _right = cast(
        tuple[np.ndarray, np.ndarray, np.ndarray],
        spl.eig(A, b=B, left=True, right=True),
    )
    _wd, _vl, _vr = _eigvals, _left, _right
    if B is None:
        _scl = np.diag(_vl.conj().T.dot(_vr))
    else:
        _scl = np.diag(_vl.conj().T.dot(B).dot(_vr))
    _sr = np.sqrt(_scl)
    _sl = _sr.conj()
    _vr = _vr / _sr.reshape(1, -1)
    _vl = _vl / _sl.reshape(1, -1)
    return _wd, _vl, _vr


def eig_low_rank(U, V):
    """Approach like Exact DMD.

    Suppose the full matrix is A = U V^T, with U, V of shape (n, r).

    We compute the eigendecomposition of the small matrix A_tilde = V^T U of shape (r, r)

    A_tilde = W @ L @ W_inv

    Then A = (U W) @ L @ (V W_inv)^H

    Furthermore, we scale the left and right eigenvectors so that they are orthonormal.
    """
    _At = V.T.dot(U)
    _w, _vl, _vr = scaled_eig(_At)
    _vl = V.dot(_vl) / _w.conj().reshape(1, -1)
    _vr = U.dot(_vr)
    return _w, _vl, _vr


def truncate_sequence(seq, order):
    """
    Truncation of scalar sequence.

    Possible order parameters

    - Float: Max value to retain
    - Integer: Keep first N values
    - 'full': Retain all pairs
    """
    _idx = np.argsort(seq)
    if isinstance(order, float):
        msk = seq[_idx] <= order
        idx = _idx[msk]
    elif isinstance(order, int):
        idx = _idx[:order]
    elif order.lower() == "full":
        idx = _idx
    else:
        raise NotImplementedError(f"Undefined threshold for order={order}")
    return idx


def make_random_matrix(Ndim, Nrnk, zrng, wrng, dt=-1):
    """
    Random (Ndim x Ndim) matrix of rank Nrnk, with randomized eigenvalues
    ranged in `zrng` and `wrng`.  If dt>0 is given, the eigenvalues will be
    mapped to discrete-time.
    The eigenpairs are always assumed to be conjugate.
    """
    _Nr = Nrnk // 2
    _U = np.random.rand(Ndim, _Nr) + 1j * np.random.rand(Ndim, _Nr)
    U0 = np.hstack([_U, _U.conj()])
    V0 = np.linalg.pinv(U0).conj().T
    z0 = np.random.rand(_Nr) * (zrng[1] - zrng[0]) + zrng[0]
    w0 = np.random.rand(_Nr) * (wrng[1] - wrng[0]) + wrng[0]
    _L = z0 + 1j * w0
    if dt > 0:
        _L = np.exp(_L * dt)
    L0 = np.hstack([_L, _L.conj()])
    A = U0.dot(np.diag(L0)).dot(V0.conj().T)
    return A, (L0, U0, V0)


def real_lowrank_from_eigpairs(
    lam: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct a real low-rank factorization V_real @ B @ U_real.T from (possibly complex)
    eigenpairs of a real matrix A = sum_i lam_i * v_i * u_i^H.

    Args:
        lam:    (r,) eigenvalues
        U:      (n, r) left eigenvectors
        V:      (n, r) right eigenvectors
        tol:    tolerance to identify real vs complex eigenvalues

    Returns:
        B:      (r_real, r_real) block-diagonal (1x1 for real, 2x2 for conjugate pairs)
        U_real: (m, r_real)
        V_real: (n, r_real)
    """
    r = lam.shape[0]

    used = np.zeros(r, dtype=bool)
    U_blocks = []
    V_blocks = []
    B_blocks = []

    scl = np.sqrt(2)
    for i in range(r):
        if used[i]:
            continue
        l = lam[i]
        v = V[:, i]
        u = U[:, i]
        if np.abs(l.imag) <= tol:  # real eigenvalue
            # 1x1 block: contribution = l * v * u^T  (real)
            U_blocks.append(np.real(u).reshape(-1, 1))
            V_blocks.append(np.real(v).reshape(-1, 1))
            B_blocks.append(np.array([[np.real(l)]]))
            used[i] = True
        else:
            # find its conjugate partner by matching values
            conj_idx = None
            for j in range(i + 1, r):
                if used[j]:
                    continue
                if np.abs(lam[j] - np.conj(l)) <= 1e-10 * (1.0 + np.abs(l)):
                    conj_idx = j
                    break
            if conj_idx is None:
                raise RuntimeError(f"Could not find conjugate partner for eigenvalue index {i}")

            # Use only the 'positive imag' representative to build a 2x2 real block
            if l.imag < 0:
                # swap to always use the positive imaginary one
                i, conj_idx = conj_idx, i
                l = lam[i]
                v = V[:, i]
                u = U[:, i]

            # For the conjugate pair sum,
            #   S_pair = l v u^T + l̄ v̄ ū^T = 2 V_block @ C @ U_block^T
            #   with C = [[a, b], [-b, a]]
            # We scale each block by sqrt(2)
            U_block = scl * np.stack([np.real(u), np.imag(u)], axis=1)
            V_block = scl * np.stack([np.real(v), np.imag(v)], axis=1)
            a, b = np.real(l), np.imag(l)
            C_block = np.array([[a, b], [-b, a]], dtype=float)

            U_blocks.append(U_block)
            V_blocks.append(V_block)
            B_blocks.append(C_block)
            used[i] = True
            used[conj_idx] = True

    # Concatenate blocks
    U_real = np.concatenate(U_blocks, axis=1)
    V_real = np.concatenate(V_blocks, axis=1)
    # Build block-diagonal B
    sizes = [B.shape[0] for B in B_blocks]
    B = np.zeros((sum(sizes), sum(sizes)))
    ofs = 0
    for Bi in B_blocks:
        k = Bi.shape[0]
        B[ofs : ofs + k, ofs : ofs + k] = Bi
        ofs += k

    return B, U_real, V_real


def mode_split(
    l: np.ndarray,
    U: np.ndarray,
    comp: str = "ri",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split eigenvalues and modes according to the requested components.

    Consider r pairs of eigenvalues and modes, modes in shape (r, n),
    this function splits them as (r, k) and (r, k, n), where k is the number of
    requested components.

    For example, if comp='ri', then k=2, and the two components are the
    real and imaginary parts of the eigenvalues and modes.

    Args:
        l:      (r,) eigenvalues
        U:      (r, n) modes
        comp:   'r' - real, 'i' - imag, 'a' - amplitude, 'p' - phase,
                can be composed like 'ri', 'ap' etc.;
                default is 'ri'.

    Returns:
        l_split: (r, k)
        U_split: (r, k, n)
    """
    r = l.shape[0]
    assert U.shape[0] == r, "Mismatch in number of modes"
    n = U.shape[1]

    assert all([c in "riap" for c in comp]), "Invalid comp string"

    l_split, U_split = [], []
    for c in comp:
        if c == "r":
            l_split.append(np.real(l).reshape(r, 1))
            U_split.append(np.real(U).reshape(r, 1, n))
        elif c == "i":
            l_split.append(np.imag(l).reshape(r, 1))
            U_split.append(np.imag(U).reshape(r, 1, n))
        elif c == "a":
            l_split.append(np.abs(l).reshape(r, 1))
            U_split.append(np.abs(U).reshape(r, 1, n))
        elif c == "p":
            l_split.append(np.angle(l).reshape(r, 1))
            U_split.append(np.angle(U).reshape(r, 1, n))
    l_split = np.hstack(l_split)
    U_split = np.concatenate(U_split, axis=1)

    return l_split, U_split


def _phiS(U: torch.Tensor, V: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """
    Compute a batch of phi_1(s_i * S) where S = V^T U, using block matrix exponentials.

    Args:
      U: (n, r)
      V: (n, r)
      s: (m,) or (m, 1) real scalars (can be any float dtype)

    Returns:
      phi: (m, r, r) with phi[i] = phi_1(s[i] * S)

    Complexity:
      One-time S = V^T U: O(n r^2); then per s_i a matrix exp of size (2r x 2r).
      This is exact and stable even if S is singular.
    """
    _, r = U.shape
    m = s.numel()

    # S = V^T U  (r x r)
    S = V.transpose(0, 1) @ U  # (r, r)

    # Batch the scaled matrices: X_i = s_i * S
    X = s.view(m, 1, 1) * S.unsqueeze(0)  # (m, r, r)

    # Build block matrices [[X_i, I],[0, 0]] of size (2r x 2r), batched over m
    Z = torch.zeros((m, 2 * r, 2 * r), dtype=U.dtype, device=U.device)
    Z[:, :r, :r] = X
    I_r = torch.eye(r, dtype=U.dtype, device=U.device).expand(m, r, r)
    Z[:, :r, r:] = I_r  # top-right block = I

    # Exponential of each block; top-right block is phi_1(X_i)
    EZ = torch.matrix_exp(Z)  # (m, 2r, 2r)
    phi = EZ[:, :r, r:]  # (m, r, r)

    return phi


def expm_low_rank(
    U: torch.Tensor, V: torch.Tensor, s: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    """
    Compute B_i = b @ exp(s_i * U V^T) for i=1..m, in batch.

    Uses the identity: exp(sA) = I + U [s * phi_1(s S)] V^T,  S = V^T U.
    So: b @ exp(sA) = b + (bU) [s * phi_1(s S)] V^T.

    Args:
      U: (n, r)
      V: (n, r)
      s: (m,) list/1D tensor of scalars
      b: (batch, n)  rows are left-multipliers

    Returns:
      out: (m, batch, n) where out[i] = b @ exp(s[i] * U V^T)
    """
    # Ensure shapes
    assert U.ndim == 2 and V.ndim == 2
    n, r = U.shape
    assert V.shape == (n, r)
    assert b.ndim == 2 and b.shape[1] == n

    device, dtype = U.device, U.dtype

    # Cast s and b to match U/V
    s = s.reshape(-1).to(device=device, dtype=dtype)  # (m,)
    b = b.to(device=device, dtype=dtype)  # (batch, n)

    # Get phi_1(s_i * S) for all s_i: (m, r, r)
    phi = _phiS(U, V, s)

    # Precompute invariants
    BU = b @ U  # (batch, r)
    Vt = V.transpose(0, 1)  # (r, n)

    # Build M_i = s_i * phi_1(s_i * S): (m, r, r)
    M = s.view(-1, 1, 1) * phi

    # (m, batch, r) = (m, 1, r, r) @ (1, batch, r, 1) style via bmm
    # Use batched matmul: (m, batch, r) = (m, batch, r) @ (m, r, r)
    BU_expanded = BU.unsqueeze(0).expand(M.shape[0], -1, -1)  # (m, batch, r)
    tmp = torch.bmm(BU_expanded, M)  # (m, batch, r)

    # Final update: (m, batch, n) = (m, batch, r) @ (r, n)
    update = torch.matmul(tmp, Vt)  # (m, batch, n)

    # Add the identity contribution b: broadcast to (m, batch, n)
    out = b.unsqueeze(0) + update
    return out


def expm_full_rank(W: torch.Tensor, s: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute B_i = b @ exp(s_i * W) for i=1..m, in batch.

    Args:
       W: (n, n) full rank matrix
       s: (m,) list/1D tensor of scalars
       b: (batch, n)  rows are left-multipliers

    Returns:
       out: (m, batch, n) where out[i] = b @ exp(s[i] * W)
    """
    assert W.ndim == 2 and W.shape[0] == W.shape[1]
    n = W.shape[0]
    assert b.ndim == 2 and b.shape[1] == n

    device, dtype = W.device, W.dtype
    s = s.reshape(-1).to(device=device, dtype=dtype)  # (m,)
    m = s.shape[0]
    b = b.to(device=device, dtype=dtype)  # (batch, n)

    # Batch compute matrix exponentials: (m, n, n)
    W_batch = s[:, None, None] * W[None, :, :]  # (m, n, n)
    expW = torch.matrix_exp(W_batch)  # (m, n, n)
    # Batch multiply: (m, batch, n) = (m, batch, n) @ (m, n, n)
    b_expanded = b.unsqueeze(0).expand(m, -1, -1)  # (m, batch, n)
    out = torch.bmm(b_expanded, expW)  # (m, batch, n)

    return out


def logm_low_rank(V: np.ndarray, U: np.ndarray, dt: float = 1.0):
    """
    Given A = V U^T dt (n x n, rank r) with V, U (n x r), compute logm(A).

    Technically, this logarithm is ill-defined if A is not full rank,
    but here we compute the following:

    Suppose A has the eigendecomposition
    A = W @ L @ R^H
    and we let
    logm(A) = W @ (log(L)/dt) @ R^H

    For computational purpose, we return two real factors of logm(A),
    with the help of real_lowrank_from_eigpairs.

    Notes:
        This approach does not work when A has negative real eigenvalues.

    Args:
        V, U : (n, r) real arrays
            Tall factors of A = V U^T. Columns need not be orthonormal.

    Returns:
        V_out, U_out : (n, r) real arrays
            Tall factors of logm(A).
    """
    wd, vl, vr = eig_low_rank(V, U)
    wc = disc2cont(wd, dt)

    wd_real = np.real(wd)
    wd_imag = np.imag(wd)
    _msk = (wd_real < 0) & (np.abs(wd_imag) <= 1e-10 * (1.0 + np.abs(wd_real)))
    if np.any(_msk):
        raise Warning(f"logm_low_rank: A has negative real eigenvalues: {wd[_msk]}.")

    B, U_real, V_out = real_lowrank_from_eigpairs(wc, vl, vr)
    U_out = U_real @ B.T

    return V_out, U_out
