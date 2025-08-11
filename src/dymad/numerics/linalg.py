import numpy as np
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
