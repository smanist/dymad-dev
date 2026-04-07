import numpy as np
import pytest
import torch

from dymad.core import build_transform_module
from dymad.numerics import Manifold, central_diff


def check_data(out, ref, label=""):
    for _s, _t in zip(out, ref, strict=False):
        assert np.allclose(_s, _t), f"{label} failed: {_s} != {_t}"
    print(f"{label} passed.")


Xs = [
    np.array([[1.0, 2.0], [1.1, 3.0], [1.2, 4.0], [1.3, 5.0], [1.4, 6.0], [1.5, 7.0]]),
    np.array([[2.2, 3.4], [2.3, 3.5], [2.4, 3.6], [2.5, 3.7]]),
]
Xn = np.array([1.32, 2.4])


def test_modes_const():
    # ----------
    # Initialize
    mktr = build_transform_module(
        [
            {"type": "scaler", "mode": "std"},
            {"type": "lift", "fobs": "poly", "Ks": [2, 3]},
            {"type": "svd", "order": 2, "ifcen": True},
        ]
    )
    mktr.fit([torch.as_tensor(item, dtype=torch.float64) for item in Xs])

    # ----------
    # Forward
    def forward(x):
        return mktr.transform([x])[0].squeeze()

    modes_f = mktr.get_forward_modes(ref=Xn)
    modes_c = central_diff(forward, Xn)
    assert np.allclose(modes_f, modes_c), f"Forward modes failed: {modes_f} != {modes_c}"

    # ----------
    # Backward
    def backward(z):
        return mktr.inverse_transform([z])[0].squeeze()

    modes_f = mktr.get_backward_modes(ref=Xn)
    modes_c = central_diff(backward, Xn).T
    assert np.allclose(modes_f, modes_c), f"Backward modes failed: {modes_f} != {modes_c}"


def run_modes_ndr(N=1000, skp=100, ndr="dm"):
    theta = np.linspace(0, 2 * np.pi * (N - 1) / N, N).reshape(-1, 1)
    x = np.hstack((2 * np.cos(theta), np.sin(theta), np.sin(theta) ** 3))

    # Embedding
    if ndr == "dm":
        trn_ndr = {
            "type": "dm",
            "edim": 3,
            "Knn": 15,
            "Kphi": 4,
            "inverse": "gmls",
            "order": 1,
            "mode": "full",
        }
    elif ndr == "isomap":
        trn_ndr = {"type": "isomap", "edim": 2, "Knn": 15, "Kphi": 4, "inverse": "gmls", "order": 1}
    trns = build_transform_module(trn_ndr)
    trns.fit([torch.as_tensor(x, dtype=torch.float64)])
    z = trns.transform([x])[0]

    # Preparation
    man = Manifold(x, 1, K=15, g=4, T=4)
    man.precompute()

    def backward(z):
        return trns.inverse_transform([z])[0].squeeze()

    def forward(x):
        return trns.transform([x])[0].squeeze()

    # Modes
    modes_fa, err_f = [], []
    modes_ba, err_b = [], []
    for idx in range(0, N, skp):
        P = man._T[idx].T.dot(man._T[idx])
        modes_fa.append(trns.get_forward_modes(ref=x[idx]))
        modes_ff = central_diff(forward, x[idx], v=P)
        err_f.append(np.linalg.norm(modes_fa[-1] - modes_ff) / np.linalg.norm(modes_fa[-1]))

        modes_ba.append(trns.get_backward_modes(ref=z[idx]))
        modes_bf = central_diff(backward, z[idx]).T
        err_b.append(np.linalg.norm(modes_ba[-1] - modes_bf) / np.linalg.norm(modes_ba[-1]))

    return (x, z), (modes_fa, err_f), (modes_ba, err_b)


@pytest.mark.parametrize("ndr", ["dm", "isomap"])
def test_modes_ndr(ndr):
    _, (_, err_f), (_, err_b) = run_modes_ndr(200, 50, ndr=ndr)

    if ndr == "isomap":
        eps_f = 10.0
    else:
        eps_f = 4e-4

    assert np.mean(err_f) < eps_f, f"{ndr} forward modes failed: {np.mean(err_f)}"
    assert np.mean(err_b) < 2e-5, f"{ndr} backward modes failed: {np.mean(err_b)}"


@pytest.mark.parametrize(
    "config",
    [
        {"type": "DiffMap", "edim": 2, "mode": "knn", "Knn": 20, "inverse": "pinv"},
        {"type": "Isomap", "edim": 2, "Knn": 20, "inverse": "pinv"},
    ],
)
def test_ndr_pinv_backward_modes_match_inverse_jacobian_transpose(config) -> None:
    tt = np.linspace(0, np.pi, 201)
    cc = np.cos(tt)
    ss = np.sin(tt)
    mixing = np.random.default_rng(1).random((2, 20))
    payload = (np.vstack([cc, ss]).T @ mixing).astype(np.float64)

    module = build_transform_module(config)
    module.fit([torch.as_tensor(payload, dtype=torch.float64)])
    latent = module.transform([payload])[0][len(tt) // 3]
    child = module.transforms[0]

    modes = module.get_backward_modes(ref=latent)

    assert modes.shape == (latent.shape[0], payload.shape[1])
    assert child._pseudo_inverse_matrix is not None
    assert child._pseudo_inverse_matrix.shape == modes.shape
    assert np.allclose(modes, child._pseudo_inverse_matrix)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ifgd = 1

    if ifgd:
        N, skp = 1000, 100
        (x, z), (modes_fa, err_f), (modes_ba, err_b) = run_modes_ndr(N, skp, ndr="dm")
        print(np.mean(err_f), np.mean(err_b))

        clrs = ["b", "r", "g"]

        def plot_modes(x, z, modes):
            if z.shape[1] == 2:
                z = np.hstack((z, np.zeros((z.shape[0], 1))))
            f = plt.figure()
            ax = f.add_subplot(111, projection="3d")
            ax.plot(x[:, 0], x[:, 1], x[:, 2], "b.")
            ax.plot(z[:, 0], z[:, 1], z[:, 2], "k.")

            for idx in range(len(modes)):
                jdx = idx * skp
                ln = np.vstack([x[jdx], z[jdx]]).T
                plt.plot(ln[0], ln[1], ln[2], "b:")

                m = modes[idx]
                if m.shape[0] == 2:
                    m = np.vstack((m, np.zeros((1, m.shape[1]))))
                if m.shape[1] == 2:
                    m = np.hstack((m, np.zeros((m.shape[0], 1))))
                for _i, _m in enumerate(m):
                    plt.quiver(*x[jdx], *_m, color=clrs[_i])
                for _i, _m in enumerate(m.T):
                    plt.quiver(*z[jdx], *_m, color=clrs[_i])
            plt.axis("equal")

        plot_modes(x, z, modes_fa)
        plot_modes(x, z, modes_ba)

        f = plt.figure()
        plt.semilogy(np.arange(0, N, skp), err_f, "bo", label="forward", markerfacecolor="none")
        plt.semilogy(np.arange(0, N, skp), err_b, "r^", label="backward", markerfacecolor="none")
        plt.xlabel("Index")
        plt.ylabel("Relative error")
        plt.legend()

    plt.show()
