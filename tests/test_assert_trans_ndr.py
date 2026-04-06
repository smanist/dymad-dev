import numpy as np
import pytest

from dymad.transform import DiffMap, DiffMapVB, Isomap

N, M = 201, 20
tt = np.linspace(0, np.pi, N)
cc = np.cos(tt)
ss = np.sin(tt)
Ms = np.random.rand(2, M)
X = np.vstack([cc, ss]).T @ Ms
nrm = np.linalg.norm(X)

opts = [
    dict(edim=2, Knn=20, inverse="gmls", order=1, Kphi=4),
    dict(edim=2, mode="full", Knn=None, inverse="gmls", order=1, Kphi=4),
    dict(edim=2, mode="knn", Knn=20, inverse="gmls", order=1, Kphi=4),
    dict(edim=2, mode="knn", Knn=20, inverse="pinv"),
    dict(edim=2, mode="knn", Knn=20, inverse="gmls", order=1, Kphi=4),
]
mdls = [Isomap, DiffMap, DiffMap, DiffMap, DiffMapVB]
lbls = ["Isomap", "DiffMap-full", "DiffMap-knn", "DiffMap-pinv", "DiffMapVB"]
epss = [3e-5, 2e-3, 3e-3, 0.4, 0.03]


def run_ndr(MDL, opt, inp):
    # First pass
    mdl = MDL(**opt)
    mdl.fit([inp])
    Zt = mdl.transform([inp])[0]
    Xr = mdl.inverse_transform([Zt])[0]

    # Reload test
    stt = mdl.state_dict()
    reld = MDL(**opt)
    reld.load_state_dict(stt)
    Zn = reld.transform([inp])[0]
    Xs = reld.inverse_transform([Zn])[0]

    return (Zt, Xr), (Zn, Xs)


@pytest.mark.parametrize("idx", range(len(mdls)))
def test_ndr(idx):
    (Zt, Xr), (Zn, Xs) = run_ndr(mdls[idx], opts[idx], X)
    assert np.linalg.norm(X - Xr) / nrm < epss[idx], f"{lbls[idx]} recon. error"
    assert np.linalg.norm(Zt - Zn) / np.linalg.norm(Zt) < 1e-13, f"{lbls[idx]} reload, transform"
    assert np.linalg.norm(Xr - Xs) / nrm < 1e-14, f"{lbls[idx]} reload, inv. trans."


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    stys = ["bo", "rs", "g^", "md", "cv"]
    f = plt.figure()
    for idx in range(len(mdls)):
        (Zt, Xr), (Zn, Xs) = run_ndr(mdls[idx], opts[idx], X)
        print(
            np.linalg.norm(X - Xr) / nrm,
            np.linalg.norm(Zt - Zn) / np.linalg.norm(Zt),
            np.linalg.norm(Xr - Xs) / nrm,
        )
        plt.plot(Zt[:, 0], Zt[:, 1], stys[idx], markerfacecolor="none", label=lbls[idx])
    plt.legend()

    plt.show()
