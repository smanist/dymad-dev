import numpy as np

def _solve_lin_sys(zs):
    _z = np.array(zs).reshape(-1)
    _m = len(zs)
    _b = np.zeros(_m,)
    _b[0] = 1.0
    _A = [np.ones_like(_b), _z]
    for _i in range(2, _m):
        _A.append(_z**_i)
    _A = np.array(_A)
    _s = np.linalg.solve(_A, _b)
    # Enforce symmetry
    return 0.5 * (_s + _s[::-1].conj())

def generate_coef(order, eps):
    _zs = 1 + 1j * (2/(order+1) * np.arange(1,order+1) - 1)
    _ts = (-_zs / (1+eps*_zs)).conj()
    if order == 1:
        return _ts, _zs, np.array([1.]), np.array([1.])
    _ts = 0.5 * (_ts + _ts[::-1].conj())  # Enforce symmetry
    _cs = _solve_lin_sys(_ts)
    _ds = _solve_lin_sys(_zs)
    return _ts, _zs, _cs, _ds

def rational_kernel(theta, order, eps):
    _ts, _zs, _cs, _ds = generate_coef(order, eps)
    _ex = np.exp(-1j*theta)
    _om = (_ex-1) / eps
    _K = 0
    for _i in range(order):
        _K += _cs[_i]/(_om-_ts[_i]) - _ds[_i]/(_om-_zs[_i])
    _K *= _ex/(eps*2*np.pi)
    return _K
