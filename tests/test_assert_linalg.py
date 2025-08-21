import numpy as np
import scipy.linalg as spl
import torch

from dymad.numerics import check_orthogonality, expm_full_rank, expm_low_rank, scaled_eig

def cmp(sol, ref):
    return np.linalg.norm(sol-ref) / np.linalg.norm(ref)

N = 4
eps = 1e-14
A = np.eye(N) + 0.1*np.random.rand(N,N)
B = np.eye(N) + 0.1*np.random.rand(N,N)
B = (B + B.T)/2

def test_scaled_eig_std():
    w, vl, vr = scaled_eig(A)
    L = np.diag(w)
    vL = vl.conj().T
    assert cmp(L.dot(vL), vL.dot(A)) <= eps   # Left eigenvectors
    assert cmp(vr.dot(L), A.dot(vr)) <= eps   # Right eigenvectors
    assert cmp(vL.dot(A).dot(vr), L) <= eps   # Diagonalization
    assert cmp(np.diag(vl.conj().T.dot(vl)), \
        np.diag(vr.conj().T.dot(vr))) <= eps  # Equalized norms
    assert check_orthogonality(vl, vr)[1] <= eps  # Orthogonality

def test_scaled_eig_gen():
    w, vl, vr = scaled_eig(A, B=B)
    L = np.diag(w)
    vL = vl.conj().T
    assert cmp(L.dot(vL).dot(B), vL.dot(A)) <= eps   # Left eigenvectors
    assert cmp(B.dot(vr).dot(L), A.dot(vr)) <= eps   # Right eigenvectors
    assert cmp(vL.dot(A).dot(vr), L) <= eps          # Diagonalization
    assert cmp(np.diag(vl.conj().T.dot(vl)), \
        np.diag(vr.conj().T.dot(vr))) <= eps         # Equalized norms
    assert check_orthogonality(vl, vr, M=B)[1] <= eps    # Orthogonality

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
    A = -np.eye(N) + 0.1*np.random.randn(N,N)
    t = np.linspace(0, 1, T)
    b = np.random.randn(B, N)

    E_ref = _eval_expm(A, t, b)
    E = expm_full_rank(torch.tensor(A), torch.tensor(t), torch.tensor(b))

    assert E.shape == (T, B, N)
    assert cmp(E.detach().cpu().numpy(), E_ref) <= eps

def test_expm_low_rank():
    T, B, N, R = 11, 6, 4, 2
    eps = 1e-15
    U = np.random.randn(N, R)
    V = np.random.randn(N, R)
    t = np.linspace(0, 1, T)
    b = np.random.randn(B, N)

    E_ref = _eval_expm(U @ V.T, t, b)
    E = expm_low_rank(torch.tensor(U), torch.tensor(V), torch.tensor(t), torch.tensor(b))

    assert E.shape == (T, B, N)
    assert cmp(E.detach().cpu().numpy(), E_ref) <= eps
