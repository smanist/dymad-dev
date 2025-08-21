import numpy as np
import scipy.linalg as spl
import torch
from typing import Tuple

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
            _I = np.argmax(np.cumsum(_s2)/np.sum(_s2) > order)
        else:
            _n, _m = X.shape
            _bt = min(_n, _m)/max(_n, _m)
            _om = 0.56*_bt**3 - 0.95*_bt**2 + 1.82*_bt + 1.43
            _I = np.argmax(_S < _om * np.median(_S))
        _Ur = _U[:,:_I]
        _Sr = _S[:_I]
        _Vr = _Vh[:_I].conj().T
    elif isinstance(order, int):
        _Ur = _U[:,:order]
        _Sr = _S[:order]
        _Vr = _Vh[:order].conj().T
    elif order.lower() == 'full':
        _Ur, _Sr, _Vr = _U, _S, _Vh.conj().T
    else:
        raise NotImplementedError(f"Undefined threshold for order={order}")
    return _Ur, _Sr, _Vr

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
    if len(v1.shape) == 1:
        # Just one vector
        return _check_direction(v1, v2)
    _, _N = v1.shape
    _d = np.zeros(_N,)
    for _i in range(_N):
        _d[_i] = _check_direction(v1[:,_i], v2[:,_i])
    return _d

def _check_direction(v1, v2):
    """
    Expecting the angle between v1 and v2 is zero, i.e., aligned.
    """
    _v1 = v1.reshape(-1)
    _v2 = v2.reshape(-1)
    _cc = _v1.conj().dot(_v2) / (np.linalg.norm(_v1)*np.linalg.norm(_v2))
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
    _err = np.mean(np.abs(_L-np.eye(_m)))
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
    _wd, _vl, _vr = spl.eig(A, b=B, left=True, right=True)
    if B is None:
        _scl = np.diag(_vl.conj().T.dot(_vr))
    else:
        _scl = np.diag(_vl.conj().T.dot(B).dot(_vr))
    _sr = np.sqrt(_scl)
    _sl = _sr.conj()
    _vr = _vr / _sr.reshape(1,-1)
    _vl = _vl / _sl.reshape(1,-1)
    return _wd, _vl, _vr

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
    elif order.lower() == 'full':
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
    _Nr = Nrnk//2
    _U = np.random.rand(Ndim, _Nr) + 1j * np.random.rand(Ndim, _Nr)
    U0 = np.hstack([_U, _U.conj()])
    V0 = np.linalg.pinv(U0).conj().T
    z0 = np.random.rand(_Nr) * (zrng[1]-zrng[0]) + zrng[0]
    w0 = np.random.rand(_Nr) * (wrng[1]-wrng[0]) + wrng[0]
    _L = z0 + 1j*w0
    if dt > 0:
        _L = np.exp(_L*dt)
    L0 = np.hstack([_L, _L.conj()])
    A  = U0.dot(np.diag(L0)).dot(V0.conj().T)
    return A, (L0, U0, V0)

def real_lowrank_from_eigpairs(
    lam: np.ndarray,          # shape (r,), complex eigenvalues
    V: np.ndarray,            # shape (n, r), right eigenvectors (columns)
    W: np.ndarray = None,     # shape (m, r), left eigenvectors (columns), optional
    normalize_biorth: bool = True,
    tol: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct a real low-rank factorization U_real @ B @ V_real.T from (possibly complex)
    eigenpairs of a real matrix A such that A ≈ sum_i lam_i * v_i * w_i^T.

    If W is None, we assume a square matrix and compute W from eigenvectors of A.T
    (you must pass the original A in that case via closure or adapt as needed).

    Returns:
        U_real: (m, r_real)
        B:      (r_real, r_real) block-diagonal (1x1 for real, 2x2 for conjugate pairs)
        V_real: (n, r_real)
    """
    r = lam.shape[0]
    n = V.shape[0]
    if W is None:
        raise ValueError("Left eigenvectors W must be provided (or compute them externally).")

    # Optional: biorthonormalize so that w_i^T v_j = delta_ij
    if normalize_biorth:
        for i in range(r):
            denom = W[:, i].T @ V[:, i]
            if np.abs(denom) < tol:
                continue
            V[:, i] = V[:, i] / denom

    used = np.zeros(r, dtype=bool)
    # Worst case: all pairs are complex -> real rank doubles; allocate generously then trim.
    U_blocks = []
    V_blocks = []
    B_blocks = []

    for i in range(r):
        if used[i]:
            continue
        λ = lam[i]
        v = V[:, i]
        w = W[:, i]
        if np.abs(λ.imag) <= tol:  # real eigenvalue
            # 1x1 block: contribution = λ * v * w^T  (real)
            # Ensure real numerically:
            U_blocks.append(np.real(w).reshape(-1, 1))
            V_blocks.append(np.real(v).reshape(-1, 1))
            B_blocks.append(np.array([[np.real(λ)]]))
            used[i] = True
        else:
            # find its conjugate partner
            # match by value; in practice you may want a more robust pairing
            conj_idx = None
            for j in range(i+1, r):
                if used[j]:
                    continue
                if np.abs(lam[j] - np.conj(λ)) <= 1e-10 * (1.0 + np.abs(λ)):
                    conj_idx = j
                    break
            if conj_idx is None:
                raise RuntimeError("Could not find conjugate partner for eigenvalue index {}".format(i))

            # Use only the 'positive imag' representative to build a 2x2 real block
            if λ.imag < 0:
                # swap to always use the positive imaginary one
                i, conj_idx = conj_idx, i
                λ = lam[i]; v = V[:, i]; w = W[:, i]

            a, b = np.real(λ), np.imag(λ)
            # Decompose vectors into real/imag parts
            p, q = np.real(v), np.imag(v)     # right vec v = p + i q
            rL, sL = np.real(w), np.imag(w)   # left vec w = rL + i sL

            # As derived: for the conjugate pair sum,
            #   S_pair = λ v w^T + λ̄ v̄ w̄^T = U_block @ C @ V_block^T
            # with U_block = [rL, sL], V_block = [p, -q],
            # and C = [[a, b], [-b, a]]
            U_block = np.stack([rL, sL], axis=1)   # shape (m, 2)
            V_block = np.stack([p, -q], axis=1)    # shape (n, 2)
            C_block = np.array([[a, b],
                                [-b, a]], dtype=float)

            U_blocks.append(U_block)
            V_blocks.append(V_block)
            B_blocks.append(C_block)
            used[i] = True
            used[conj_idx] = True

    # Concatenate blocks
    U_real = np.concatenate(U_blocks, axis=1) if U_blocks else np.zeros((W.shape[0], 0))
    V_real = np.concatenate(V_blocks, axis=1) if V_blocks else np.zeros((V.shape[0], 0))
    # Build block-diagonal B
    sizes = [B.shape[0] for B in B_blocks]
    B = np.zeros((sum(sizes), sum(sizes)))
    ofs = 0
    for Bi in B_blocks:
        k = Bi.shape[0]
        B[ofs:ofs+k, ofs:ofs+k] = Bi
        ofs += k

    return U_real, B, V_real

def reconstruct_from_real_blocks(U_real: np.ndarray, B: np.ndarray, V_real: np.ndarray) -> np.ndarray:
    """A = U_real @ B @ V_real.T"""
    return U_real @ B @ V_real.T

def _phiS(U: torch.Tensor, V: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """
    Compute a batch of phi_1(s_i * S) where S = V^T U, using block matrix exponentials.
    Inputs:
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
    Z = torch.zeros((m, 2*r, 2*r), dtype=U.dtype, device=U.device)
    Z[:, :r, :r] = X
    I_r = torch.eye(r, dtype=U.dtype, device=U.device).expand(m, r, r)
    Z[:, :r, r:] = I_r  # top-right block = I

    # Exponential of each block; top-right block is phi_1(X_i)
    EZ = torch.matrix_exp(Z)              # (m, 2r, 2r)
    phi = EZ[:, :r, r:]                   # (m, r, r)

    return phi

def expm_low_rank(U: torch.Tensor,
                  V: torch.Tensor,
                  s: torch.Tensor,
                  b: torch.Tensor) -> torch.Tensor:
    """
    Compute B_i = b @ exp(s_i * U V^T) for i=1..m, in batch.

    Uses the identity: exp(sA) = I + U [s * phi_1(s S)] V^T,  S = V^T U.
    So: b @ exp(sA) = b + (bU) [s * phi_1(s S)] V^T.

    Inputs:
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
    s = s.reshape(-1).to(device=device, dtype=dtype)   # (m,)
    b = b.to(device=device, dtype=dtype)               # (batch, n)

    # Get phi_1(s_i * S) for all s_i: (m, r, r)
    phi = _phiS(U, V, s)

    # Precompute invariants
    BU = b @ U                                        # (batch, r)
    Vt = V.transpose(0, 1)                            # (r, n)

    # Build M_i = s_i * phi_1(s_i * S): (m, r, r)
    M = s.view(-1, 1, 1) * phi

    # (m, batch, r) = (m, 1, r, r) @ (1, batch, r, 1) style via bmm
    # Use batched matmul: (m, batch, r) = (m, batch, r) @ (m, r, r)
    BU_expanded = BU.unsqueeze(0).expand(M.shape[0], -1, -1)  # (m, batch, r)
    tmp = torch.bmm(BU_expanded, M)                           # (m, batch, r)

    # Final update: (m, batch, n) = (m, batch, r) @ (r, n)
    update = torch.matmul(tmp, Vt)                            # (m, batch, n)

    # Add the identity contribution b: broadcast to (m, batch, n)
    out = b.unsqueeze(0) + update
    return out

def expm_full_rank(W: torch.Tensor,
                   s: torch.Tensor,
                   b: torch.Tensor) -> torch.Tensor:
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
    s = s.reshape(-1).to(device=device, dtype=dtype)   # (m,)
    m = s.shape[0]
    b = b.to(device=device, dtype=dtype)               # (batch, n)

    # Batch compute matrix exponentials: (m, n, n)
    W_batch = s[:, None, None] * W[None, :, :]         # (m, n, n)
    expW = torch.matrix_exp(W_batch)                   # (m, n, n)
    # Batch multiply: (m, batch, n) = (m, batch, n) @ (m, n, n)
    b_expanded = b.unsqueeze(0).expand(m, -1, -1)      # (m, batch, n)
    out = torch.bmm(b_expanded, expW)                  # (m, batch, n)

    return out
