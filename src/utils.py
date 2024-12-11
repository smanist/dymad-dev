import jax.numpy as jn
import jax
import matplotlib.pyplot as plt
import numpy as np
from scipy import special as spc

# ----------------------
# Model construction
# ----------------------

def init_mat_kaiming(shape, key):
    # Treat first dimension as input
    return jax.random.normal(shape=shape, key=jax.random.PRNGKey(key)) * jn.sqrt(2/shape[0])

def init_mat_xavier(shape, key):
    return jax.random.normal(shape=shape, key=jax.random.PRNGKey(key)) * jn.sqrt(2/(shape[0]+shape[1]))

# ----------------------
# Sampling
# ----------------------

def sample_unif_2d(xrng, Nx):
    x01 = np.linspace(xrng[0][0], xrng[0][1], Nx[0])
    x02 = np.linspace(xrng[1][0], xrng[1][1], Nx[1])
    X1, X2 = np.meshgrid(x01, x02)
    x0s = np.vstack([X1.reshape(-1), X2.reshape(-1)]).T
    return x0s

# ----------------------
# Plot data
# ----------------------

def plt_data(dat, sty='-', ts=None, idx=None, fig=None):
    Nt, Nv = dat[0].shape
    if ts is None:
        ts = np.arange(Nt)
    if idx is None:
        idx = np.arange(Nv)
        Nr = Nv
    else:
        Nr = len(idx)
    if fig is None:
        f, ax = plt.subplots(nrows=Nr, sharex=True)
    else:
        f, ax = fig
    for _t in dat:
        for _i in range(Nr):
            ax[_i].plot(ts, _t[:,idx[_i]], sty)
    return f, ax

def plt_hist(hist, keys, stys, avr=1, lbls=None, fig=None, ifnrm=False):
    if keys == 'all':
        keys = np.sort([_k for _k in hist.keys()])
    Nk = len(keys)
    if fig is None:
        fig = plt.figure()
    else:
        plt.figure(fig.number)
    if ifnrm:
        _, _l = _makeSegments(hist[keys[0]], avr)
        scl = np.max(_l)
    else:
        scl = 1.0
    for _k in range(Nk):
        if lbls is None:
            lbl = keys[_k]
        else:
            lbl = lbls[_k]
        _n, _l = _makeSegments(hist[keys[_k]], avr)
        plt.semilogy(_n, _l/scl, stys[_k], label=lbl)
    plt.legend()
    return fig

def _makeSegments(segs, avr):
    _l = np.hstack(segs)
    _N = len(_l)
    _n = np.arange(_N)
    if avr > 1:
        _M = (_N//avr)*avr
        _l = _l[:_M].reshape(-1,avr).mean(axis=1)
        _n = _n[:_M:avr]
    return _n, _l

# ----------------------
# Numerical integration
# ----------------------

_NC = {
    '1' : [0.5, 0.5],          # Trapezoid
    '2' : [1./3, 4./3, 1./3],  # Simpson 1/3
    '3' : [3./8, 9./8, 9./8, 3./8],  # Simpson 3/8
    '4' : [14./45, 64./45, 24./45, 64./45, 14./45],  # Boole
}
def weight_nc(N, order, dt):
    if order == 0:
        _W = np.ones(N,)
        _W[-1] = 0.0
        return jn.array(_W*dt)
    assert (N-1) % order == 0  # Check required number of samples
    _w = _NC[str(order)]
    _W = np.zeros(N,)
    _N = (N-1) // order
    for _i in range(_N):
        _W[_i*order:(_i+1)*order+1] += _w
    return jn.array(_W*dt)

# ----------------------
# Backward difference
# ----------------------

_BDF = {
    '1' : [-1., 1.],
    '2' : [1./2, -2., 3./2],
    '3' : [-1./3, 3./2, -3., 11./6],
    '4' : [1./4, -4./3, 3., -4., 25./12],
}
def weight_bdf(order):
    return jn.array(_BDF[str(order)])

# ------------------------
# Jacobi(1,1)
# ------------------------

def get_base(order, coo):
    _P = np.zeros((order, len(coo)))
    for _i in range(order):
        _P[_i] = spc.eval_jacobi(_i, 1, 1, coo)
    return _P

def get_der1(order, coo):
    _P = np.zeros((order, len(coo)))
    for _i in range(1,order):
        _P[_i] = (_i+3)/2 * spc.eval_jacobi(_i-1, 2, 2, coo)
    return _P