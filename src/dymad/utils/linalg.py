import numpy as np

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
