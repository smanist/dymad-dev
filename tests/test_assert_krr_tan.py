import numpy as np
import torch

from dymad.modules import make_krr
from dymad.numerics import DMF, Manifold, ManifoldAnalytical, tangent_2torus

# Data
a, b = 1.0, 2.0
TEST_SEED = 0


def torus_sample(Nsmp, *, seed=TEST_SEED):
    tmp = np.random.default_rng(seed).random((2, Nsmp)) * 2 * np.pi
    x = (a * np.cos(tmp[0]) + b) * np.cos(tmp[1])
    y = (a * np.cos(tmp[0]) + b) * np.sin(tmp[1])
    z = a * np.sin(tmp[0])
    tar = np.vstack([x, y, z]).T
    return tar


X = torus_sample(1000)

T = tangent_2torus(X, b)
F = np.vstack([X[:, 0] ** 2 / 4 + X[:, 1] ** 2, X[:, 1] * X[:, 2]]).T
Y = np.einsum("ij,ijk->ik", F, T)

Ntrn = 500
Xtrn = X[:Ntrn]
Ytrn = Y[:Ntrn]
Ttrn = T[:Ntrn]
Xtst = X[Ntrn:]
Ytst = Y[Ntrn:]
Ttst = T[Ntrn:]

# KRR Options
RIDGE = 1e-10
opt_rbf = {"type": "sc_rbf", "input_dim": 3, "lengthscale_init": 1.0}
opt_dm = {
    "type": "sc_dm",
    "input_dim": 3,
    "eps_init": None,
}
opt_opk = {"type": "op_tan", "input_dim": 3, "output_dim": 3, "kopts": opt_rbf}

opt_share = {"type": "share", "kernel": opt_rbf, "dtype": torch.float64, "ridge_init": RIDGE}
opt_dmshr = {"type": "share", "kernel": opt_dm, "dtype": torch.float64, "ridge_init": RIDGE}
opt_tange = {"type": "tangent", "kernel": opt_opk, "dtype": torch.float64, "ridge_init": RIDGE}
opts = [
    opt_share,  # vanilla KRR
    opt_dmshr,  # KRR with DM
    opt_tange,  # with estimated tangent, comparable to vanilla KRR, unnecessarily better
    opt_tange,  # with analytical manifold, nearly perfect tangent, slightly lower mean error
]
lbls = ["RBF", "DM", "Tangent", "Tan-Ana", "DMF"]


def run_krr():
    prds = []
    for i, opt in enumerate(opts):
        _krr = make_krr(**opt)
        _krr.set_train_data(Xtrn, Ytrn)
        if i == 2:
            _man = Manifold(Xtrn, d=2, T=4)
            _man.precompute()
            _krr.set_manifold(_man)
        elif i == 3:
            _man = ManifoldAnalytical(Xtrn, d=2, fT=lambda x: tangent_2torus(x, b))
            _man.precompute()
            _krr.set_manifold(_man)
        _krr.fit()
        with torch.no_grad():
            Yprd = _krr(torch.tensor(Xtst, dtype=torch.float64)).cpu().numpy()
            prds.append(Yprd)
    return prds


def run_dm():
    dm = DMF(None, n_neighbors=None, alpha=1)
    dm.fit_krr(Xtrn, Ytrn, ridge=RIDGE)
    Yprd = dm.predict_krr(Xtst)
    return Yprd


def check_tangent(Y, T):
    _Y = Y[..., None]
    _R = _Y - np.matmul(np.swapaxes(T, -1, -2), np.matmul(T, _Y))
    _R = _R.squeeze()
    res = np.linalg.norm(_R, axis=1) / np.linalg.norm(Y, axis=1)
    return res


def check_error(prd, tru):
    ref = np.linalg.norm(tru, axis=1)
    ref[ref < 1e-3] = 1.0
    err = np.linalg.norm(tru - prd, axis=1) / ref
    return err


def test_krr():
    prds = run_krr()
    _p = run_dm()
    prds.append(_p)

    errs = [check_error(_p, Ytst) for _p in prds]
    ress = [check_tangent(_p, Ttst) for _p in prds]

    assert errs[0].mean() < 5e-4, "KRR share, error"
    assert errs[1].mean() < 4e-5, "KRR, DM, error"
    assert errs[2].mean() < 0.02, "KRR tangent, estimate T, error"
    assert errs[3].mean() < 5e-4, "KRR tangent, analytical T, error"
    assert errs[4].mean() < 4e-5, "DMF, error"

    assert ress[0].mean() < 4e-5, "KRR share, tangent residual"
    assert ress[1].mean() < 1e-5, "KRR, DM, tangent residual"
    assert ress[2].mean() < 0.02, "KRR tangent, estimate T, tangent residual"
    assert ress[3].mean() < 1e-15, "KRR tangent, analytical T, tangent residual"
    assert ress[4].mean() < 1e-5, "DMF, tangent residual"

    dif = np.linalg.norm(prds[1] - prds[4]) / np.linalg.norm(Ytst)
    assert dif < 1e-5, "DMF, difference"


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def plot3d(X, Y, scl=1, fig=None, sty="k-"):
        N = len(X)

        if fig is None:
            f = plt.figure()
            ax = f.add_subplot(projection="3d")
        else:
            f, ax = fig
        ax.plot(X[:, 0], X[:, 1], X[:, 2], "b.", markersize=1)
        for _i in range(N):
            _p = X[_i] + scl * Y[_i]
            _c = np.vstack([X[_i], _p]).T
            ax.plot(_c[0], _c[1], _c[2], sty)
        return f, ax

    prds = run_krr()
    _p = run_dm()
    prds.append(_p)

    dif = np.linalg.norm(prds[1] - prds[4]) / np.linalg.norm(Ytst)
    print("DMF vs DM, relative diff:", dif)

    Nprd = len(prds)

    fig = plot3d(Xtst, Ytst, scl=0.2, fig=None, sty="k-")
    fig = plot3d(Xtst, prds[0], scl=0.2, fig=fig, sty="r--")
    fig = plot3d(Xtst, prds[1], scl=0.2, fig=fig, sty="g:")
    fig = plot3d(Xtst, prds[2], scl=0.2, fig=fig, sty="b:")
    fig = plot3d(Xtst, prds[3], scl=0.2, fig=fig, sty="c:")
    fig = plot3d(Xtst, prds[4], scl=0.2, fig=fig, sty="m:")

    stys = ["bo", "r^", "gs", "md", "cx"]

    f = plt.figure()
    for i in range(Nprd):
        err = check_error(prds[i], Ytst)
        print(np.mean(err), np.max(err))
        plt.semilogy(err, stys[i], markerfacecolor="none", label=lbls[i])
    plt.legend()

    f = plt.figure()
    for i in range(Nprd):
        res = check_tangent(prds[i], Ttst)
        print(np.mean(res), np.max(res))
        plt.semilogy(res, stys[i], markerfacecolor="none", label=lbls[i])
    plt.legend()

    plt.show()
