import numpy as np

from dymad.numerics import check_orthogonality, scaled_eig

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
