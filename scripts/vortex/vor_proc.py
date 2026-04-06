import matplotlib.pyplot as plt
import numpy as np

from dymad.io import DataInterface
from dymad.utils import compare_contour, plot_contour

config_path = "vor_model.yaml"
Nx, Ny = 199, 449

ifdat = 0
ifmod = 1

if ifdat:
    dat = np.load("./data/raw.npz")["vor"]
    Nt, Nx, Ny = dat.shape
    ts = np.arange(Nt)
    dt = 1.0
    X = dat.reshape(Nt, -1)

    Nspl = 140
    ttrn = ts[:Nspl]
    Xtrn = X[:Nspl]
    ttst = ts[Nspl:]
    Xtst = X[Nspl:]

    np.savez_compressed("data/cylinder.npz", x=Xtrn, t=ttrn)
    np.savez_compressed("data/test.npz", x=Xtst, t=ttst)

if ifmod:
    ifemb = 0
    ifrec = 0
    ifsvd = 0
    ifbck = 0
    iffor = 0
    ifcor = 1

    dat = np.load("data/cylinder.npz")
    ttrn = dat["t"]
    Xtrn = dat["x"]
    dt = ttrn[1] - ttrn[0]

    dat = np.load("data/test.npz")
    Xtst = dat["x"]

    trn_svd = {"type": "svd", "ifcen": True, "order": 0.9999}
    trn_dmf = {
        "type": "dm",
        "edim": 3,
        "Knn": 15,
        "Kphi": 3,
        "inverse": "gmls",
        "order": 1,
        "mode": "full",
    }

    di = DataInterface(config_path=config_path, config_mod={"transform_x": [trn_svd, trn_dmf]})
    Zsvd = di.encode(Xtst, rng=[0, 1])
    Zdmf = di.encode(Zsvd, rng=[1, 2])
    Xsvd = di.decode(Zdmf, rng=[1, 2])
    Xrec = di.decode(Xsvd, rng=[0, 1])

    idx = 5
    ref = Xtst[idx].reshape(1, Nx, Ny)
    dXrf = (Xtst[idx + 1] - Xtst[idx - 1]) / (2 * dt)
    dXrf = dXrf.reshape(1, Nx, Ny)
    dZrf = (Zdmf[idx + 1] - Zdmf[idx - 1]) / (2 * dt)
    modes_b = di.get_backward_modes(ref=Zdmf[idx]).reshape(-1, Nx, Ny)
    modes_f = di.get_forward_modes(ref=Xtst[idx]).reshape(-1, Nx, Ny)

    if ifemb:
        Ztrn = di.encode(Xtrn)
        f = plt.figure()
        for _i in range(3):
            plt.plot(ttrn, Ztrn[:, _i], label=f"DM coord {_i + 1}")
        plt.xlabel("t")
        plt.ylabel("z")
        plt.legend()

    if ifrec:
        x_tru = Xtst[idx].reshape(Nx, Ny)
        x_rec = Xrec[idx].reshape(Nx, Ny)
        f, ax = compare_contour(x_tru, x_rec, vmin=-4, vmax=4, figsize=(12, 2))
        for _a in ax:
            _a.set_axis_off()

    if ifsvd:
        modes_s = di.get_backward_modes(ref=Xsvd[idx], rng=[0, 1]).reshape(-1, Nx, Ny)
        arrays = np.concatenate([ref, 200 * modes_s[:5]], axis=0)
        labels = [f"step {idx}"] + [f"mode {_i + 1}" for _i in range(5)]
        f, ax = plot_contour(
            arrays, figsize=(12, 4), colorbar=True, label=labels, grid=(2, 3), mode="contourf"
        )
        for _a in ax.flatten():
            _a.set_axis_off()

    if ifbck:
        arrays = np.concatenate([ref, modes_b], axis=0)
        labels = [f"step {idx}"] + [f"mode {_i + 1}" for _i in range(3)]
        f, ax = plot_contour(
            arrays,
            vmin=-4,
            vmax=4,
            figsize=(8, 4),
            colorbar=True,
            label=labels,
            grid=(2, 2),
            mode="contourf",
        )
        for _a in ax.flatten():
            _a.set_axis_off()

        dXdt = np.sum(dZrf[:, None, None] * modes_b, axis=0, keepdims=True)
        f, ax = compare_contour(dXrf[0], dXdt[0], figsize=(12, 2))
        for _a in ax:
            _a.set_axis_off()

    if iffor:
        arrays = np.concatenate([ref, 40000 * modes_f], axis=0)
        labels = [f"step {idx}"] + [f"mode {_i + 1}" for _i in range(3)]
        f, ax = plot_contour(
            arrays,
            vmin=-4,
            vmax=4,
            figsize=(8, 4),
            colorbar=True,
            label=labels,
            grid=(2, 2),
            mode="contourf",
        )
        for _a in ax.flatten():
            _a.set_axis_off()

        dZdt = np.sum(dXrf * modes_f, axis=(1, 2))

        f = plt.figure()
        labels = ["1", "2", "3"]
        values = np.stack([dZdt, dZrf], axis=1)  # shape (3,2)
        x = np.arange(len(labels))
        width = 0.35
        plt.bar(x - width / 2, values[:, 0], width, label="Estimate", color="blue")
        plt.bar(x + width / 2, values[:, 1], width, label="Finite Diff.", color="orange")
        plt.xticks(x, labels)
        plt.legend()
        plt.ylabel("Rate")

    if ifcor:
        _f = modes_f.reshape(3, -1)
        _b = modes_b.reshape(3, -1)

        f, ax = plt.subplots(1, 3, figsize=(9, 3))
        im0 = ax[0].imshow(_f @ _b.T, vmin=-1, vmax=1, cmap="bwr")
        ax[0].set_title("dz/dx * dx/dz")
        plt.colorbar(im0, ax=ax[0])

        im1 = ax[1].imshow(_f @ _f.T, cmap="bwr")
        ax[1].set_title("dz/dx * (dz/dx)^T")
        plt.colorbar(im1, ax=ax[1])

        im2 = ax[2].imshow(_b @ _b.T, cmap="bwr")
        ax[2].set_title("(dx/dz)^T * dx/dz")
        plt.colorbar(im2, ax=ax[2])

plt.show()
