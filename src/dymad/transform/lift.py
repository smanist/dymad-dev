import numpy as np

# --------------------
# Elemental components, assuming row input/output for easier indexing
# i.e., output is (n_feature, n_sample)
# --------------------
## Single state
def psi_fourier(x, K):
    _x = x.reshape(-1)
    _p = [np.ones_like(_x)]
    for _k in K:
        _p.append(np.cos((_k+1)*_x))
        _p.append(np.sin((_k+1)*_x))
    return np.vstack(_p)

def psi_fourier_inverse(p, K):
    assert K > 0, "Inversion only applies when K>=1"
    _c, _s = p[1], p[2]
    return np.arctan2(_s, _c)

def psi_monomial(x, K):
    if isinstance(K, int):
        return x.reshape(-1)**K
    _x = x.reshape(-1)
    return np.vstack([_x**k for k in K])

def psi_monomial_inverse(p, K):
    assert K > 0, "Inversion only applies when K>=1"
    return p[1]

## two-state
def psi_polar(x, K):
    assert x.shape[0] == 2, "Input must be (2,N) or (,2)"

    _r = np.linalg.norm(x, axis=0)
    _t = np.arctan2(x[1], x[0])

    _pr = psi_monomial(_r, np.arange(K[0]))
    _pt = psi_fourier(_t, np.arange(K[1]))
    return _pr, _pt

def psi_polar_inverse(p, K):
    """Inverse of psi_polar."""
    assert K[0] >= 1 and K[1] >= 1, "Inversion only applies when K_r>=1 and K_t>=1"
    _r = p[0][1]
    _c, _s = p[1][1:3]
    _x = _r*_c
    _y = _r*_s
    return _x, _y

PSI_MAP = {
    'f': [psi_fourier, psi_fourier_inverse],
    'm': [psi_monomial, psi_monomial_inverse],
    'p': [psi_polar, psi_polar_inverse]
}

# --------------------
# Assembly components, assuming column input/output for interface with training
# i.e., output is (n_sample, n_feature)
# --------------------
## Cross-product of psi's
def _cross(ps):
    """Cross product of multiple observables from elemental components.

    Inputs k arrays of (n_feature, n_sample) - the ith array:
        [
        [pi1(x1), pi1(x2), ..., pi1(xN)]
        ...
        [pim(x1), pim(x2), ..., pim(xN)]]

    Outputs the cross product of the features, with shape (N, P1*P2*...*Pk),
    where Pi is the number of features in the ith array.

    Also returns the list of number of features in each component.
    """
    _Nx, _Np = ps[0].shape[1], len(ps)
    _p1 = ps[0].T
    for _i in range(1, _Np):
        _p1 = np.einsum('ij,ik->ijk', _p1, ps[_i].T).reshape(_Nx, -1)
    _Ns = [_p.shape[0] for _p in ps]
    return _p1

def poly_cross(x, Ks):
    assert x.shape[1] == len(Ks), "Input dimension does not match the number of powers"
    _Np = len(Ks)
    _ps = []
    for i in range(_Np):
        _ps.append(psi_monomial(x[:,i], np.arange(Ks[i])))
    return _cross(_ps)

def poly_inverse(Z, Ks):
    """Inverse of poly_cross, given the poly order in each component."""
    assert Z.shape[1] == np.prod(Ks), "Input dimension does not match the number of features"
    _Nx = Z.shape[0]
    _Z = Z.reshape(_Nx, *Ks)
    # Sequentially access _Z[:, 1, 0, ...], _Z[:, 0, 1, ...], etc.
    _Np = len(Ks)
    _out = []
    for i in range(_Np):
        idx = [0] * _Np
        idx[i] = 1
        _out.append(_Z[(slice(None), *idx)])
    return np.stack(_out, axis=1)

def _collect_index(opts):
    _idx = []
    for _opt in opts:
        if isinstance(_opt[0], list):
            _idx.extend(_opt[0])
        elif isinstance(_opt[0], int):
            _idx.append(_opt[0])
    return _idx

def _get_slice(z, i, N):
    idx = [0] * N
    idx[i] = slice(None)
    return z[(slice(None), *idx)].T

def mixed_cross(x, opts):
    """
    opts = list of tuples (index, type, K)

    Examples:
        Monomials: (0, 'm', K)
        Fourier:   (1, 'f', K)
        Polar:     ([0,1], 'p', (Kr, Kt))
    """
    _idx = _collect_index(opts)
    assert len(_idx) == len(set(_idx)), "Indices in opts should not overlap"

    _ps = [[] for _ in range(max(_idx)+1)]
    for _i, _t, _k in opts:
        if _t not in PSI_MAP:
            raise ValueError(f"Unknown observable type {_t}")
        if isinstance(_i, list):
            _x = np.vstack([x[:,_j] for _j in _i])
            _p = PSI_MAP[_t][0](_x, _k)
            for _j, _ij in enumerate(_i):
                _ps[_ij] = _p[_j]
        else:
            _ps[_i] = PSI_MAP[_t][0](x[:,_i], np.arange(_k))
    return _cross(_ps)

def mixed_inverse(Z, opts):
    """Inverse of mixed_cross, given the options."""
    _idx = _collect_index(opts)
    Ks = np.zeros((len(_idx),), dtype=int)
    for _i, _t, _k in opts:
        if _t == 'm':
            Ks[_i] = _k
        elif _t == 'f':
            Ks[_i] = 2*_k+1
        elif _t == 'p':
            Ks[_i[0]] = _k[0]
            Ks[_i[1]] = 2*_k[1]+1

    assert Z.shape[1] == np.prod(Ks), "Input dimension does not match the number of features"
    _No = max(_idx)+1
    _Nx = Z.shape[0]
    _Z = Z.reshape(_Nx, *Ks)

    _out = np.zeros((len(Ks), _Nx))
    for _i, _t, _k in opts:
        if _t in 'mf':
            _out[_i] = PSI_MAP[_t][1](_get_slice(_Z, _i, _No), _k)
        elif _t == 'p':
            _r = _get_slice(_Z, _i[0], _No)
            _t = _get_slice(_Z, _i[1], _No)
            _x1, _x2 = psi_polar_inverse([_r, _t], _k)
            _out[_i[0]] = _x1
            _out[_i[1]] = _x2
        else:
            raise ValueError(f"Unknown observable type {_t}")
    return _out.T
