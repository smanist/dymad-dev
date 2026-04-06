import matplotlib.pyplot as plt
import numpy as np
import torch

from dymad.io import load_model
from dymad.models import DKBF
from dymad.numerics import complex_plot, scaled_eig
from dymad.sako import SpectralAnalysis
from dymad.training import LinearTrainer
from dymad.utils import TrajectorySampler, plot_trajectory

B = 64
N = 21
t_grid = np.linspace(0, 10, N)
dt = t_grid[1] - t_grid[0]

## Identity
# _T = np.eye(2)
# _S = np.eye(2)
# ## Rotation
# _t = np.pi/4
# _c, _s = np.cos(_t), np.sin(_t)
# _T = np.array([[_c, _s], [-_s, _c]])
# _S = _T.T
## Shear
_t = 0.2
_T = np.array([[1, _t], [0, 1]])
_S = np.array([[1, -_t], [0, 1]])

mu = -0.5
lm = -3


def f(t, x):
    _y = _T.dot(x)
    _d = np.array([mu * _y[0], lm * (_y[1] - _y[0] ** 2)])
    return _S.dot(_d)


Jac = _S.dot(np.diag([mu, lm])).dot(_T)

w0 = np.array([mu, lm]) + 1j * 0
w0 = np.hstack([w0, 2 * w0[0], 2 * w0[1], w0[0] + w0[1]])
wa = np.exp(w0 * dt)

mdl_kl = {
    "name": "kp_model",
    "encoder_layers": 0,
    "decoder_layers": 0,
    "koopman_dimension": 9,
    "activation": "none",
    "weight_init": "xavier_uniform",
    "predictor_type": "exp",
}
trn_kl = [{"type": "lift", "fobs": "poly", "Ks": [3, 3]}]

ref = {
    "n_epochs": 1,
    "save_interval": 1,
    "load_checkpoint": False,
}
trn_ln = {
    "method": "full",
}
trn_ln.update(ref)
trn_tr = {
    "method": "truncated",
    "params": 0.999,
}
trn_tr.update(ref)
trn_sa = {
    "method": "sako",
    "params": 4,
    "kwargs": {"remove_one": True},
}
trn_sa.update(ref)

config_path = "kp_model.yaml"

cfgs = [
    ("dkbf_ln", DKBF, LinearTrainer, {"model": mdl_kl, "transform_x": trn_kl, "training": trn_ln}),
    ("dkbf_tr", DKBF, LinearTrainer, {"model": mdl_kl, "transform_x": trn_kl, "training": trn_tr}),
    ("dkbf_sa", DKBF, LinearTrainer, {"model": mdl_kl, "transform_x": trn_kl, "training": trn_sa}),
]

IDX = [0, 1, 2]
# IDX = [1]
labels = [cfgs[i][0] for i in IDX]

ifdat = 0
iftrn = 1
ifprd = 1
ifint = 1

if ifdat:
    sampler = TrajectorySampler(f, config="kp_data.yaml")
    ts, xs, ys = sampler.sample(t_grid, batch=B, save="./data/kp.npz")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    for i in range(B):
        ax.plot(xs[i, :, 0], xs[i, :, 1], alpha=0.5)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("2D Trajectories")
    plt.tight_layout()

    fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    for i in range(B):
        axs[0].plot(ts[i], xs[i, :, 0], alpha=0.5)
        axs[1].plot(ts[i], xs[i, :, 1], alpha=0.5)
    axs[0].set_ylabel("x1")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("x2")
    plt.tight_layout()

if iftrn:
    for i in IDX:
        mdl, MDL, Trainer, opt = cfgs[i]
        opt["model"]["name"] = f"kp_{mdl}"
        trainer = Trainer(config_path, MDL, config_mod=opt)
        trainer.train()

if ifprd:
    sampler = TrajectorySampler(f, config="kp_data.yaml")
    ts, xs, ys = sampler.sample(t_grid, batch=1)
    x_data = xs[0]
    t_data = ts[0]

    res = [x_data]
    for i in IDX:
        mdl, MDL, _, _ = cfgs[i]
        _, prd_func = load_model(MDL, f"kp_{mdl}.pt")
        with torch.no_grad():
            pred = prd_func(x_data, t_data)
        res.append(pred)

    plot_trajectory(np.array(res), t_data, "KP", labels=["Truth"] + labels, ifclose=False)

if ifint:
    saln = SpectralAnalysis(DKBF, "kp_dkbf_ln.pt", dt=dt, reps=1e-10, etol=None)
    satr = SpectralAnalysis(DKBF, "kp_dkbf_tr.pt", dt=dt, reps=1e-10, etol=None)
    sasa = SpectralAnalysis(DKBF, "kp_dkbf_sa.pt", dt=dt, reps=1e-10, etol=None)

    sas = [saln, satr, sasa]
    lbs = ["DT-LN", "DT-TR", "DT-SA"]

    ifprd = 1
    ifeig, ifeic, ifpsp, ifres = 1, 1, 1, 0
    ifspe, ifegf, iftrn = 1, 1, 1

    if ifprd:
        J = 16
        sampler = TrajectorySampler(f, config="kp_data.yaml")
        ts, xs, _ = sampler.sample(t_grid, batch=J)
        x0s = xs[:, 0, :].squeeze()

        for _i in range(3):
            sas[_i].plot_pred(x0s, ts[0], ref=xs, idx="all", figsize=(6, 8), title=lbs[_i])

    if ifeig:
        ## Eigenvalues
        MRK = 15
        fig, ax = plt.subplots(ncols=3, sharey=True, figsize=(15, 5))
        for _i in range(3):
            fig, ax[_i], _ls = sas[_i].plot_eigs(fig=(fig, ax[_i]))
            (_l,) = ax[_i].plot(wa.real, wa.imag, "kx", markersize=MRK)
            ax[_i].set_title(f"{lbs[_i]}\nMax res: {sas[_i]._res[-1]:4.3e}")
            ax[_i].legend(_ls + [_l], [lbs[_i], "Filtered", "Truth"], loc=1)

        if ifpsp:
            # Pseudospectra
            xs = np.linspace(-1.3, 1.3, 51)
            gg = np.vstack([xs, xs])
            rng = np.array([0.1, 0.25])

            # Predicted
            pss, psk = [], []
            for _s in sas:
                grid, _pss = _s.estimate_ps(gg, mode="disc", method="standard", return_vec=False)
                grid, _psk = _s.estimate_ps(gg, mode="disc", method="sako", return_vec=False)
                pss.append((grid, _pss))
                psk.append((grid, _psk))

            for _i in range(3):
                grid, _pss = pss[_i]
                grid, _psk = psk[_i]
                f, ax[_i] = complex_plot(
                    grid, 1 / _pss, rng, fig=(f, ax[_i]), mode="line", lwid=2, lsty="dotted"
                )
                f, ax[_i] = complex_plot(grid, 1 / _psk, rng, fig=(f, ax[_i]), mode="line", lwid=1)

        for _i in range(2):
            ax[_i].set_xlim([-0.4, 1.3])
            ax[_i].set_ylim([-0.7, 0.7])

    if ifeic:
        ## Eigenvalues
        MRK = 15
        fig, ax = plt.subplots(ncols=3, sharey=True, figsize=(15, 5))
        for _i in range(3):
            fig, ax[_i], _ls = sas[_i].plot_eigs(fig=(fig, ax[_i]), mode="cont")
            (_l,) = ax[_i].plot(w0.real, w0.imag, "kx", markersize=MRK)
            ax[_i].set_title(f"{lbs[_i]}\nMax res: {sas[_i]._res[-1]:4.3e}")
            ax[_i].legend(_ls + [_l], [lbs[_i], "Filtered", "Truth"], loc=1)

        if ifpsp:
            # Pseudospectra
            zs = np.linspace(-8.0, 0.5, 51)
            ws = np.linspace(-4.0, 4.0, 51)
            gg = np.vstack([zs, ws])
            rng = np.array([0.25, 0.5])
            # Predicted
            pss, psk = [], []
            for _s in sas:
                grid, _pss = _s.estimate_ps(gg, mode="cont", method="standard", return_vec=False)
                grid, _psk = _s.estimate_ps(gg, mode="cont", method="sako", return_vec=False)
                pss.append((grid, _pss))
                psk.append((grid, _psk))

            for _i in range(3):
                grid, _pss = pss[_i]
                grid, _psk = psk[_i]
                fig, ax[_i] = complex_plot(
                    grid, 1 / _pss, rng, fig=(fig, ax[_i]), mode="line", lwid=2, lsty="dotted"
                )
                fig, ax[_i] = complex_plot(
                    grid, 1 / _psk, rng, fig=(fig, ax[_i]), mode="line", lwid=1
                )

        for _i in range(2):
            ax[_i].set_xlim([-8.0, 0.5])
            ax[_i].set_ylim([-4.0, 4.0])

    if ifres:
        ## Residuals
        stys = ["bo", "r^", "gs", "md", "c*"]
        fig, ax = plt.subplots()
        for _i in range(3):
            ax.semilogy(
                np.abs(sas[_i]._wd), sas[_i]._res, stys[_i], label=lbs[_i], markerfacecolor="none"
            )
        ax.set_xlabel("Norm of eigenvalue")
        ax.set_ylabel("Residual")
        ax.legend()

    if ifspe:
        # Spectral measure
        stys = ["b-", "r-", "g--", "m--", "c-"]

        def func_obs(x):
            _x1, _x2 = x.T
            return _x1 + _x2

        vgs = []
        for _s in sas:
            _t, _v = _s.estimate_measure(func_obs, 6, 0.1, thetas=501)
            vgs.append((_t, _v))

        _arg = np.angle(wa)
        _amp = np.max(vgs[0][1])

        fig = plt.figure()
        for _i in range(3):
            th, vg = vgs[_i]
            plt.plot(th, vg, stys[_i], label=lbs[_i], markerfacecolor="none")
        plt.plot([_arg[0], _arg[0]], [0, _amp], "k:", label="System frequency")
        for _a in _arg[1:]:
            plt.plot([_a, _a], [0, _amp], "k:")
        plt.legend()
        plt.xlabel("Angle, rad")
        plt.ylabel("Spectral measure")

    if ifegf:
        ## Eigenfunctions
        rngs = [[-1.5, 1.5], [-1.5, 1.5]]
        Ns = [101, 101]
        md = "real"

        fig, ax = plt.subplots(nrows=3, ncols=4, sharex=True, sharey=True, figsize=(10, 10))
        for i in range(3):
            _n = min(sas[i]._Nrank, 4)
            sas[i].plot_eigfun_2d(rngs, Ns, _n, mode=md, fig=(fig, ax[i]))
            ax[i][0].set_ylabel(lbs[i])

    if iftrn:
        sa = sasa
        ## Conjugacy map
        sa.set_conj_map(Jac)
        ## Sample trajectories
        sampler = TrajectorySampler(f, config="kp_data.yaml")
        _, sol, _ = sampler.sample(t_grid, batch=64)
        ## Plotting
        stys = ["b-", "r-", "b--", "r--"]
        f, ax = plt.subplots(ncols=3, figsize=(10, 5))
        for _i, _s in enumerate(sol):
            ax[0].plot(_s[:, 0], _s[:, 1], stys[_i % 4])
            r0, r1 = sa.mapto_cnj(_s).real.T
            ax[1].plot(r0, r1, stys[_i % 4])
            r0, r1 = sa.mapto_nrm(_s).real.T
            ax[2].plot(r0, r1, stys[_i % 4])
        ## Slow manifold
        yy = np.linspace(-1, 1, 41)
        s0 = _S.dot(
            np.array([yy, yy**2])
        ).T  # Closed-form solution is in transformed coord; so transform back
        ax[0].plot(s0[:, 0], s0[:, 1], "k-")
        r0, r1 = sa.mapto_cnj(s0).real.T
        ax[1].plot(r0, r1, "k-")
        r0, r1 = sa.mapto_nrm(s0).real.T
        ax[2].plot(r0, r1, "k-")
        ## Linear basis
        _, _, _Jr = scaled_eig(Jac)
        for _i in range(2):
            for _j in range(2):
                ax[_i].plot([0, _Jr[0, _j]], [0, _Jr[1, _j]], "go-")
        ax[2].plot([0, 0, 1], [1, 0, 0], "go-")

        ## Annotations
        ax[0].set_xlabel(r"$x_1$")
        ax[0].set_ylabel(r"$x_2$")
        ax[0].set_title("Physical space")
        ax[0].set_aspect("equal")
        ax[1].set_xlabel(r"$y_1$")
        ax[1].set_ylabel(r"$y_2$")
        ax[1].set_title('"Flatten" space')
        ax[1].set_aspect("equal")
        ax[2].set_xlabel(r"$y_1^*$")
        ax[2].set_ylabel(r"$y_2^*$")
        ax[2].set_title("Orthogonalized space")
        ax[2].set_aspect("equal")

plt.show()
