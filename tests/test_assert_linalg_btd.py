import numpy as np
import torch

from dymad.numerics.block_tridiag import (
    factorize_btd_lu,
    factorize_btd_spd,
    solve_btd_lu,
    solve_btd_spd,
)


def relerr(a: torch.Tensor, b: torch.Tensor) -> float:
    num = torch.linalg.norm((a - b).reshape(-1))
    den = torch.linalg.norm(b.reshape(-1)) + 1e-12
    return float(num / den)


def np_btd_to_dense(D: np.ndarray, U: np.ndarray, L: np.ndarray) -> np.ndarray:
    T, n = D.shape[-3], D.shape[-1]
    A = np.zeros((T * n, T * n), dtype=D.dtype)
    for k in range(T):
        i0, i1 = k * n, (k + 1) * n
        A[i0:i1, i0:i1] = D[k]
        if k < T - 1:
            j0, j1 = (k + 1) * n, (k + 2) * n
            A[i0:i1, j0:j1] = U[k]
            A[j0:j1, i0:i1] = L[k]
    return A


def numpy_solve_reference(D: torch.Tensor, U: torch.Tensor, L: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    D_np = D.detach().cpu().numpy()
    U_np = U.detach().cpu().numpy()
    L_np = L.detach().cpu().numpy()
    b_np = b.detach().cpu().numpy()

    T, n = D_np.shape[-3], D_np.shape[-1]
    batch_shape = D_np.shape[:-3]
    is_vector = b_np.ndim == len(batch_shape) + 2
    b_mat = b_np[..., None] if is_vector else b_np
    m = b_mat.shape[-1]

    batch_size = int(np.prod(batch_shape)) if batch_shape else 1
    Df = D_np.reshape(batch_size, T, n, n)
    Uf = U_np.reshape(batch_size, T - 1, n, n)
    Lf = L_np.reshape(batch_size, T - 1, n, n)
    bf = b_mat.reshape(batch_size, T, n, m)

    x_all = np.empty_like(bf)
    for i in range(batch_size):
        A = np_btd_to_dense(Df[i], Uf[i], Lf[i])
        rhs = bf[i].reshape(T * n, m)
        x = np.linalg.solve(A, rhs)
        x_all[i] = x.reshape(T, n, m)

    x_all = x_all.reshape(*batch_shape, T, n, m)
    if is_vector:
        x_all = np.squeeze(x_all, axis=-1)
    return torch.from_numpy(x_all).to(dtype=b.dtype, device=b.device)


def make_sym_spd_btd(T: int, n: int, damping: float, m_rhs: int | None = None, batch: int | None = None):
    dtype = torch.float64
    batch_shape = () if batch is None else (batch,)
    eye = torch.eye(n, dtype=dtype).expand(*batch_shape, n, n)

    D = torch.empty(*batch_shape, T, n, n, dtype=dtype)
    U = torch.empty(*batch_shape, T - 1, n, n, dtype=dtype)
    L = torch.empty(*batch_shape, T - 1, n, n, dtype=dtype)

    def rand_spd():
        R = torch.randn(*batch_shape, n, n, dtype=dtype)
        return R @ R.transpose(-1, -2) + (n + damping) * eye

    S_prev = rand_spd()
    D[..., 0, :, :] = S_prev

    for k in range(T - 1):
        Uk = 0.15 * torch.randn(*batch_shape, n, n, dtype=dtype)
        U[..., k, :, :] = Uk
        L[..., k, :, :] = Uk.transpose(-1, -2)
        Wk = torch.linalg.solve(S_prev, Uk)
        S_next = rand_spd()
        D[..., k + 1, :, :] = S_next + Uk.transpose(-1, -2) @ Wk
        S_prev = S_next

    if m_rhs is None:
        b = torch.randn(*batch_shape, T, n, dtype=dtype)
    else:
        b = torch.randn(*batch_shape, T, n, m_rhs, dtype=dtype)
    return D, U, L, b


def make_nonsym_btd(T: int, n: int, damping: float, m_rhs: int | None = None):
    dtype = torch.float64
    eye = torch.eye(n, dtype=dtype)

    D = torch.empty(T, n, n, dtype=dtype)
    U = 0.1 * torch.randn(T - 1, n, n, dtype=dtype)
    L = torch.empty(T - 1, n, n, dtype=dtype)

    def rand_inv():
        R = torch.randn(n, n, dtype=dtype)
        return R @ R.transpose(-1, -2) + (n + damping) * eye

    D_tilde_prev = rand_inv()
    D[0] = D_tilde_prev

    for k in range(T - 1):
        Mk = 0.1 * torch.randn(n, n, dtype=dtype)
        L[k] = Mk @ D_tilde_prev
        D_tilde_next = rand_inv()
        D[k + 1] = D_tilde_next + Mk @ U[k]
        D_tilde_prev = D_tilde_next

    if m_rhs is None:
        b = torch.randn(T, n, dtype=dtype)
    else:
        b = torch.randn(T, n, m_rhs, dtype=dtype)
    return D, U, L, b


def test_spd_single_rhs_dense_vs_lu_vs_spd():
    torch.manual_seed(0)
    D, U, L, b = make_sym_spd_btd(T=8, n=4, damping=1.0)

    x_dense = numpy_solve_reference(D, U, L, b)
    x_lu = solve_btd_lu(factorize_btd_lu(D, U, L), b)
    x_spd = solve_btd_spd(factorize_btd_spd(D, U), b, U)

    assert relerr(x_lu, x_dense) < 1e-15
    assert relerr(x_spd, x_dense) < 1e-15


def test_spd_multi_rhs_reuse_factorization():
    torch.manual_seed(0)
    D, U, L, b = make_sym_spd_btd(T=7, n=3, damping=1.0, m_rhs=4)

    lu_factors = factorize_btd_lu(D, U, L)
    spd_factors = factorize_btd_spd(D, U)

    x_dense = numpy_solve_reference(D, U, L, b)
    x_lu = solve_btd_lu(lu_factors, b)
    x_spd = solve_btd_spd(spd_factors, b, U)

    assert x_lu.shape == b.shape
    assert x_spd.shape == b.shape
    assert relerr(x_lu, x_dense) < 1e-15
    assert relerr(x_spd, x_dense) < 1e-15


def test_nonsym_multi_rhs_dense_vs_lu():
    torch.manual_seed(0)
    D, U, L, b = make_nonsym_btd(T=9, n=4, damping=1.0, m_rhs=3)

    x_dense = numpy_solve_reference(D, U, L, b)
    x_lu = solve_btd_lu(factorize_btd_lu(D, U, L), b)

    assert relerr(x_lu, x_dense) < 1e-15


def test_batched_spd_multi_rhs_dense_vs_spd():
    torch.manual_seed(0)
    D, U, L, b = make_sym_spd_btd(T=6, n=3, damping=1.0, m_rhs=5, batch=3)

    x_dense = numpy_solve_reference(D, U, L, b)
    spd_factors = factorize_btd_spd(D, U)
    x_spd = solve_btd_spd(spd_factors, b, U)

    assert x_spd.shape == b.shape
    assert relerr(x_spd, x_dense) < 1e-15
