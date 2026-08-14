import numpy as np
import scipy.linalg as spl
import torch

from dymad.numerics import (
    check_orthogonality,
    eig_low_rank,
    expm_full_rank,
    expm_low_rank,
    logm_low_rank,
    randomized_svd,
    real_lowrank_from_eigpairs,
    scaled_eig,
    truncated_svd,
)
from dymad.training.ls_update import _ls_full


def cmp(sol, ref):
    return np.linalg.norm(sol - ref) / np.linalg.norm(ref)


N = 4
eps = 1e-13
A = np.eye(N) + 0.1 * np.random.rand(N, N)
B = np.eye(N) + 0.1 * np.random.rand(N, N)
B = (B + B.T) / 2


def check_subspace(U1, U2):
    _s = np.linalg.svd(U1.T.dot(U2))[1]
    return 1 - _s.min()


def test_svd():
    np.random.seed(42)

    As = [np.random.rand(5, 100) for _ in range(5)]

    def LD(i):
        return As[i]

    Aa = np.vstack(As)

    u, Sref, vh = np.linalg.svd(Aa, full_matrices=False)
    Uref = u[:, :5]
    Vref = vh[:5].T

    Utrn, Strn, Vtrn = truncated_svd(Aa, 5)
    assert np.allclose(Strn, Sref[:5])
    assert np.allclose(Utrn, Uref)
    assert np.allclose(Vtrn, Vref)

    Uran, Sran, Vran = randomized_svd(LD, 5, 5, oversample=10, n_iter=2, return_u=True)
    assert np.linalg.norm(Sref[:5] - Sran) < 0.01
    assert check_subspace(Uran, Uref) < 0.005
    assert check_subspace(Vran, Vref) < 0.01

    # Reference singular values:
    # [25.1554028   4.24122589  3.93344336  3.78680366  3.73375361]
    # [25.1554028   4.23772365  3.93123059  3.77921143  3.73049082]


def test_full_least_squares_uses_source_dtype_rank_cutoff():
    singular_values = np.array([1.0, 0.5, 1.0e-8], dtype=np.float32)
    A = np.diag(singular_values)
    b = np.array([[1.0], [0.5], [1.0e-8]], dtype=np.float32)

    solution = _ls_full(A, b)

    assert np.allclose(A @ solution, b, atol=2.0e-8)
    assert np.linalg.norm(solution) < 1.5


def test_scaled_eig_std():
    w, vl, vr = scaled_eig(A)
    L = np.diag(w)
    vL = vl.conj().T
    assert cmp(L.dot(vL), vL.dot(A)) <= eps  # Left eigenvectors
    assert cmp(vr.dot(L), A.dot(vr)) <= eps  # Right eigenvectors
    assert cmp(vL.dot(A).dot(vr), L) <= eps  # Diagonalization
    assert cmp(np.diag(vL.dot(vl)), np.diag(vr.conj().T.dot(vr))) <= eps  # Equalized norms
    assert check_orthogonality(vl, vr)[1] <= eps  # Orthogonality


def test_scaled_eig_gen():
    w, vl, vr = scaled_eig(A, B=B)
    L = np.diag(w)
    vL = vl.conj().T
    assert cmp(L.dot(vL).dot(B), vL.dot(A)) <= eps  # Left eigenvectors
    assert cmp(B.dot(vr).dot(L), A.dot(vr)) <= eps  # Right eigenvectors
    assert cmp(vL.dot(A).dot(vr), L) <= eps  # Diagonalization
    assert cmp(np.diag(vL.dot(vl)), np.diag(vr.conj().T.dot(vr))) <= eps  # Equalized norms
    assert check_orthogonality(vl, vr, M=B)[1] <= eps  # Orthogonality


def test_eig_low_rank():
    N, R = 10, 3
    eps = 1e-14
    U = np.random.rand(N, R)
    V = np.random.rand(N, R)
    A = U @ V.T

    w, vl, vr = eig_low_rank(U, V)
    A_rec = vr @ np.diag(w) @ vl.conj().T

    assert cmp(A, A_rec) <= eps
    assert check_orthogonality(vl, vr)[1] <= eps


def test_real_lowrank_real():
    S = A + A.T
    w, vl, vr = scaled_eig(S)
    L, U, V = real_lowrank_from_eigpairs(w, vl, vr)
    assert cmp(S, V @ L @ U.T) <= eps


def test_real_lowrank_cplx():
    S = A - 0.9 * A.T  # Nearly skew-symmetric to force complex eigenvalues
    w, vl, vr = scaled_eig(S)
    assert np.iscomplexobj(w)
    L, U, V = real_lowrank_from_eigpairs(w, vl, vr)
    assert cmp(S, V @ L @ U.T) <= eps


def test_real_lowrank_larger():
    N = 100
    S = np.eye(N) + 0.1 * np.random.rand(N, N)
    w, vl, vr = scaled_eig(S)
    L, U, V = real_lowrank_from_eigpairs(w, vl, vr)
    assert cmp(S, V @ L @ U.T) <= 10 * eps


def _eval_expm(A, t, b):
    """
    A: (n, n)
    t: (m,)
    b: (b, n)

    Evaluates b * exp(A*t), and results in (m, b, n)
    """
    m = t.shape[0]
    E = np.empty((m, b.shape[0], b.shape[1]))
    for i in range(m):
        E[i] = b @ spl.expm(A * t[i])
    return E


def test_expm_full_rank():
    T, B, N = 11, 6, 4
    eps = 1e-15
    A = -np.eye(N) + 0.1 * np.random.randn(N, N)
    t = np.linspace(0, 1, T)
    b = np.random.randn(B, N)

    E_ref = _eval_expm(A, t, b)
    E = expm_full_rank(torch.tensor(A), torch.tensor(t), torch.tensor(b))

    assert E.shape == (T, B, N)
    assert cmp(E.detach().cpu().numpy(), E_ref) <= eps


def test_expm_low_rank():
    T, B, N, R = 11, 6, 4, 2
    eps = 4e-15
    U = np.random.randn(N, R)
    V = np.random.randn(N, R)
    t = np.linspace(0, 1, T)
    b = np.random.randn(B, N)

    E_ref = _eval_expm(U @ V.T, t, b)
    E = expm_low_rank(torch.tensor(U), torch.tensor(V), torch.tensor(t), torch.tensor(b))

    assert E.shape == (T, B, N)
    assert cmp(E.detach().cpu().numpy(), E_ref) <= eps


def test_logm_lowrank():
    N, R = 10, 3
    dt = 0.5
    eps = 1e-14

    # Make up some reasonable low-rank A = V U^T:
    # S: A nearly skew-symmetric to force complex eigenvalues
    # Q: A random orthonormal matrix
    A = np.eye(R) + 0.1 * np.random.rand(R, R)
    S = A - 0.9 * A.T
    Q, _ = np.linalg.qr(np.random.randn(N, N))

    U = Q[:, :R]
    V = U @ S

    # Use logm_low_rank
    U_log, V_log = logm_low_rank(U, V, dt=dt)

    # Exponential back using the definition in logm_low_rank
    # But this is different from standard expm since we only use part of the eigendecomposition
    wc, vl, vr = eig_low_rank(U_log, V_log)
    wd = np.exp(wc * dt)
    A_ref = np.real(vr @ np.diag(wd) @ vl.conj().T)

    assert cmp(A_ref, U @ V.T) <= eps
