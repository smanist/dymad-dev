import copy

import numpy as np
import torch

from dymad.modules import make_krr


# Data
def func(inp):
    x = np.atleast_2d(inp)
    y1 = x[:, 0] ** 2 + np.sin(x[:, 1])
    y2 = 0.5 * y1
    return np.vstack([y1, y2]).T.squeeze()


xx = np.linspace(0, 1, 6)
X, Y = np.meshgrid(xx, xx)
Xtrn = np.vstack([X.reshape(-1), Y.reshape(-1)]).T
Ytrn = func(Xtrn)
Ns = 21
s = np.linspace(0, 1, Ns)
S, T = np.meshgrid(s, s)
Xtst = np.vstack([S.reshape(-1), T.reshape(-1)]).T
Ytst = func(Xtst)

# KRR Options
RIDGE = 1e-10
opt_rbf1 = {"type": "sc_rbf", "input_dim": 2, "lengthscale_init": 10.0}
opt_rbf2 = copy.deepcopy(opt_rbf1)
opt_rbf2["lengthscale_init"] = 5.0
opt_opk1 = {
    "type": "op_sep",
    "input_dim": 2,
    "output_dim": 2,
    "kopts": [opt_rbf1],
    "Ls": np.array([[[1, 0], [0.5, 0]]]),
}
opt_opk2 = copy.deepcopy(opt_opk1)
opt_opk2["kopts"] = [opt_rbf1, opt_rbf1]
opt_opk2["Ls"] = np.array([[[1, 0], [0, 0]], [[0, 0], [0, 1]]])

opt_share = {
    "type": "share",
    "kernel": opt_rbf1,
    "dtype": torch.float64,
    "ridge_init": RIDGE,
}  # Baseline
opt_indp1 = {
    "type": "indep",
    "kernel": [opt_rbf1, opt_rbf1],
    "dtype": torch.float64,
    "ridge_init": RIDGE,
}  # Should be nearly identical to Baseline
opt_indp2 = {
    "type": "indep",
    "kernel": [opt_rbf1, opt_rbf2],
    "dtype": torch.float64,
    "ridge_init": RIDGE,
}  # Some difference in the second component
opt_opva1 = {
    "type": "opval",
    "kernel": opt_opk1,
    "dtype": torch.float64,
    "ridge_init": RIDGE,
}  # Should be close to Baseline
opt_opva2 = {
    "type": "opval",
    "kernel": opt_opk2,
    "dtype": torch.float64,
    "ridge_init": RIDGE,
}  # Should be nearly identical to Baseline
opts = [opt_share, opt_indp1, opt_indp2, opt_opva1, opt_opva2]


def run_krr():
    prds = []
    for opt in opts:
        _krr = make_krr(**opt)
        _krr.set_train_data(Xtrn, Ytrn)
        _krr.fit()
        with torch.no_grad():
            Yprd = _krr(Xtst).cpu().numpy()
            prds.append(Yprd)
    return prds


def test_krr():
    prds = run_krr()
    ref = np.linalg.norm(Ytst, axis=1)
    ref[ref < 1e-3] = 1.0
    for Yprd in prds:
        err = np.linalg.norm(Ytst - Yprd, axis=1) / ref
        assert np.mean(err) < 0.0005

    assert np.linalg.norm(prds[0] - prds[1]) < 2e-9
    assert np.linalg.norm(prds[0][:, 0] - prds[1][:, 0]) < 2e-9
    assert np.linalg.norm(prds[0] - prds[3]) < 2e-5
    assert np.linalg.norm(prds[0] - prds[4]) < 2e-9


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    prds = run_krr()
    print(np.linalg.norm(prds[0] - prds[1]))
    print(np.linalg.norm(prds[0][:, 0] - prds[1][:, 0]))
    print(np.linalg.norm(prds[0] - prds[3]))
    print(np.linalg.norm(prds[0] - prds[4]))

    mdls = ["Shared", "Indep1", "Indep2", "OpVal1", "OpVal2"]
    stys = ["bo", "rs", "g^", "mv", "cx"]
    Nkrr = len(prds)

    # Mostly overlap, except Indep2 with slightly higher errors in some locations
    f, ax = plt.subplots()
    for m_idx, Yprd_m in enumerate(prds):
        err = np.linalg.norm(Ytst - Yprd_m, axis=1) / np.linalg.norm(Ytst, axis=1)
        print(mdls[m_idx], np.linalg.norm(Ytst - Yprd_m))
        ax.semilogy(err, stys[m_idx], label=mdls[m_idx], markerfacecolor="none")
    ax.legend()

    # Visually should be very similar
    _prds = [Ytst] + prds
    _mdls = ["Truth"] + mdls
    f, ax = plt.subplots(nrows=Nkrr + 1, ncols=4, sharex=True, sharey=True, figsize=(12, 2 * Nkrr))
    for m_idx, Yprd_m in enumerate(_prds):
        yt0 = Ytst[:, 0].reshape(Ns, Ns)
        yp0 = Yprd_m[:, 0].reshape(Ns, Ns)
        yt1 = Ytst[:, 1].reshape(Ns, Ns)
        yp1 = Yprd_m[:, 1].reshape(Ns, Ns)

        cs1 = ax[m_idx, 0].contourf(S, T, yp0)
        cs2 = ax[m_idx, 1].contourf(S, T, np.abs(yt0 - yp0))
        ax[m_idx, 2].contourf(S, T, yp1, levels=cs1.levels)
        ax[m_idx, 3].contourf(S, T, np.abs(yt1 - yp1), levels=cs2.levels)

        ax[m_idx, 0].set_ylabel(_mdls[m_idx])
        plt.colorbar(
            ax[m_idx, 0].collections[0],
            ax=ax[m_idx],
            orientation="vertical",
            fraction=0.025,
            pad=0.04,
        )

    ax[0, 0].plot(Xtrn[:, 0], Xtrn[:, 1], "wo", markerfacecolor="none")
    ax[0, 0].set_title("y1")
    ax[0, 1].set_title("y1 error")
    ax[0, 2].set_title("y2")
    ax[0, 3].set_title("y2 error")

    plt.show()
