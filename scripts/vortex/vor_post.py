import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as spl

from dymad.models import DKBF, KBF
from dymad.numerics import complex_plot
from dymad.sako import SpectralAnalysis
from dymad.utils import animate, setup_logging

Nx, Ny = 199, 449
dat = np.load("data/cylinder.npz")
t_grid = dat["t"]
xs = dat["x"]
dt = t_grid[1] - t_grid[0]

_tmp = xs - np.mean(xs, axis=0)
u, _, _ = spl.svd(_tmp, full_matrices=False)
_tmp = u[:, 0] - np.mean(u[:, 0])
sp = np.fft.fft(_tmp)
fr = np.fft.fftfreq(len(t_grid)) / dt * (2 * np.pi)
ii = np.argmax(np.abs(sp))
w0 = np.abs(fr[ii])
wc = np.array([-1, 1]) * (1j * w0)
wa = np.exp(wc * dt)

sact = SpectralAnalysis(KBF, "kp_kbf_node.pt", dt=dt, reps=1e-10, etol=1e-12, remove_one=False)
sand = SpectralAnalysis(DKBF, "kp_dkbf_ln.pt", dt=dt, reps=1e-10, etol=1e-12, remove_one=False)
saae = SpectralAnalysis(DKBF, "kp_dkbf_ae.pt", dt=dt, reps=1e-10, etol=1e-12, remove_one=False)
sadm = SpectralAnalysis(DKBF, "kp_dkbf_dm.pt", dt=dt, reps=1e-10, etol=1e-12, remove_one=False)

sas = [sact, sand, saae, sadm]
lbs = ["CT-ND", "DT-LN", "DT-AE", "DT-DM"]
Nsa = len(sas)

ifprd = 1
ifeig = 1
ifmod = 1
ifani = 1

if ifprd:
    x0s = xs[0]
    for _i in range(Nsa):
        sas[_i].plot_pred(
            x0s,
            t_grid,
            ref=xs,
            idx="all",
            figsize=(6, 8),
            title=lbs[_i],
            ifobs=True,
        )

if ifeig:
    ## Eigenvalues
    MRK = 15
    f, ax = plt.subplots(ncols=Nsa, sharey=True, figsize=(15, 5))
    for _i in range(Nsa):
        f, ax[_i], _ls = sas[_i].plot_eigs(fig=(f, ax[_i]))
        (_l,) = ax[_i].plot(wa.real, wa.imag, "kx", markersize=MRK)
        ax[_i].set_title(f"{lbs[_i]}\nMax res: {sas[_i]._res[-1]:4.3e}")
        ax[_i].legend(_ls + [_l], [lbs[_i], "Filtered", "Truth"], loc=1)

    # Pseudospectra
    _x = np.linspace(-1.3, 1.3, 51)
    gg = np.vstack([_x, _x])
    rng = np.array([0.1, 0.25])

    # Predicted
    pss, psk = [], []
    for _s in sas:
        grid, _pss = _s.estimate_ps(gg, mode="disc", method="standard", return_vec=False)
        grid, _psk = _s.estimate_ps(gg, mode="disc", method="sako", return_vec=False)
        pss.append((grid, _pss))
        psk.append((grid, _psk))

    for _i in range(Nsa):
        grid, _pss = pss[_i]
        grid, _psk = psk[_i]
        f, ax[_i] = complex_plot(
            grid, 1 / _pss, rng, fig=(f, ax[_i]), mode="line", lwid=2, lsty="dotted"
        )
        f, ax[_i] = complex_plot(grid, 1 / _psk, rng, fig=(f, ax[_i]), mode="line", lwid=1)

if ifmod:
    ## Jacobian of eigenfunctions
    ## i.e., modes mapping observables to Koopman space
    for i in range(2):
        (f, ax), _ = sas[i].plot_eigjac_contour(
            eig="mode",
            lam="dt",
            comp="riap",
            idx=[0, 2, 4, 6, 8],
            shape=(Nx, Ny),
            contour_args={"figsize": (12, 8), "colorbar": True, "grid": (5, 4), "mode": "contourf"},
        )
        for _a in ax.flatten():
            _a.set_axis_off()

if ifani:
    IDX = 3
    eig = "func"

    fig, ax = plt.subplots(nrows=3, ncols=5, figsize=(16, 4))

    def contour_fig(j, eig):
        colorbar = j == 0
        cs = ax[1, 0].contourf(xs[j].reshape(Nx, Ny), vmin=-16, vmax=16, levels=20)
        ax[1, 0].set_title(f"Step {j}")
        if colorbar:
            plt.colorbar(cs, ax=ax[1, 0])
        _, _ = sas[IDX].plot_eigjac_contour(
            ref=xs[j],
            eig=eig,
            lam="dt",
            comp="ria",
            idx="all",
            shape=(Nx, Ny),
            contour_args={
                "axes": (fig, ax[:, 1:4]),
                "levels": 20,
                "colorbar": colorbar,
                "mode": "contourf",
            },
        )
        _, _ = sas[IDX].plot_eigjac_contour(
            ref=xs[j],
            eig=eig,
            lam="dt",
            comp="p",
            idx="all",
            shape=(Nx, Ny),
            contour_args={
                "axes": (fig, ax[:, 4].reshape(-1, 1)),
                "levels": 20,
                "colorbar": colorbar,
                "mode": "contourf",
            },
        )
        for _a in ax.flatten():
            _a.set_axis_off()
        return fig, ax

    # contour_fig(0)

    setup_logging()
    animate(
        lambda i: contour_fig(i, eig),
        filename=f"vis_{lbs[IDX]}_{eig}.mp4",
        fps=10,
        n_frames=len(t_grid),
    )

plt.show()
