import numpy as np

from dymad.transform import SVD, AddOne, Compose, DelayEmbedder, Identity, Scaler, make_transform


def check_data(out, ref, label=""):
    for _s, _t in zip(out, ref, strict=False):
        assert np.allclose(_s, _t), f"{label} failed: {_s} != {_t}"
    print(f"{label} passed.")


Xs = [
    np.array([[1.0, 2.0], [1.1, 3.0], [1.2, 4.0], [1.3, 5.0], [1.4, 6.0], [1.5, 7.0]]),
    np.array([[2.2, 3.4], [2.3, 3.5], [2.4, 3.6], [2.5, 3.7]]),
    np.array([[1.0, 2.5], [1.2, 4.5], [1.4, 6.5], [1.6, 8.5]]),
]
Xn = np.array([[1.32, 2.4], [1.33, 3.5], [1.34, 4.6], [1.35, 5.7]])


def test_addone():
    addo = AddOne()
    Xt = addo.transform(Xs)
    Xr = [np.concatenate([x, np.ones((len(x), 1))], axis=-1) for x in Xs]
    check_data(Xt, Xr, label="AddOne")

    Xi = addo.inverse_transform(Xt)
    check_data(Xi, Xs, label="Inverse AddOne")


def test_identity():
    iden = Identity()
    Xt = iden.transform(Xs)
    check_data(Xt, Xs, label="Identity")

    Xi = iden.inverse_transform(Xt)
    check_data(Xi, Xs, label="Inverse Identity")


def test_scaler():
    sclr = Scaler(mode="01")
    sclr.fit(Xs)
    Xt = sclr.transform([Xn])[0]

    tmp = np.vstack(Xs)
    mx, mn = np.max(tmp, axis=0), np.min(tmp, axis=0)
    Xr = (Xn - mn) / (mx - mn)
    check_data(Xt, Xr, label="Scalar 01")

    Xi = sclr.inverse_transform([Xt])[0]
    check_data(Xi, Xn, label="Inverse Scalar 01")


def test_delay():
    dely = DelayEmbedder(delay=2)
    dely.fit(Xs)
    Xt = dely.transform([Xn])[0]

    Xr = np.vstack([Xn[:3].reshape(1, -1), Xn[1:4].reshape(1, -1)])
    check_data(Xt, Xr, label="Delay")

    Xi = dely.inverse_transform([Xt])[0]
    check_data(Xi, Xn, label="Inverse Delay")


def test_svd():
    svd = SVD(order=2, ifcen=True)
    svd.fit(Xs)
    Xt = svd.transform([Xn])[0]

    tmp = np.vstack(Xs)
    avr = np.mean(tmp, axis=0)
    tmp = tmp - avr
    _, _, Vh = np.linalg.svd(tmp, full_matrices=False)
    Xr = (Xn - avr).dot(Vh.T)
    check_data(Xt, Xr, label="SVD")

    Xi = svd.inverse_transform([Xt])[0]
    check_data(Xi, Xn, label="Inverse SVD")

    # ----------
    # Loading and reloading
    dct = svd.state_dict()

    reld = SVD()
    reld.load_state_dict(dct)

    Xt = reld.transform([Xn])[0]
    check_data(Xt, Xr, label="SVD reload")

    Xi = reld.inverse_transform([Xt])[0]
    check_data(Xi, Xn, label="Inverse SVD reload")


def test_compose():
    # ----------
    # Direct construction
    cmps = Compose([Scaler(mode="std"), DelayEmbedder(delay=1)])
    cmps.fit(Xs)
    Xt = cmps.transform([Xn])[0]

    tmp = np.vstack(Xs)
    avr, std = np.mean(tmp, axis=0), np.std(tmp, axis=0)
    tmp = (Xn - avr) / std
    Xr = np.vstack([tmp[:2].reshape(1, -1), tmp[1:3].reshape(1, -1), tmp[2:4].reshape(1, -1)])
    check_data(Xt, Xr, label="Compose")

    Xi = cmps.inverse_transform([Xt])[0]
    check_data(Xi, Xn, label="Inverse Compose")

    # ----------
    # Interface
    mktr = make_transform([{"type": "scaler", "mode": "std"}, {"type": "delay", "delay": 1}])
    mktr.fit(Xs)
    Xt = mktr.transform([Xn])[0]
    check_data(Xt, Xr, label="Compose make_transform")

    Xi = mktr.inverse_transform([Xt])[0]
    check_data(Xi, Xn, label="Inverse Compose make_transform")

    # ----------
    # Loading and reloading
    dct = mktr.state_dict()

    reld = Compose()
    reld.load_state_dict(dct)

    Xt = reld.transform([Xn])[0]
    check_data(Xt, Xr, label="Compose reload")

    Xi = reld.inverse_transform([Xt])[0]
    check_data(Xi, Xn, label="Inverse Compose reload")


def test_compose_rng():
    mktr = make_transform([{"type": "scaler", "mode": "01"}, {"type": "delay", "delay": 2}])
    mktr.fit(Xs)

    # Range 0-1
    Xt = mktr.transform([Xn], rng=[0, 1])[0]

    tmp = np.vstack(Xs)
    mx, mn = np.max(tmp, axis=0), np.min(tmp, axis=0)
    Xr = (Xn - mn) / (mx - mn)
    check_data(Xt, Xr, label="Compose rng 0-1")

    Xi = mktr.inverse_transform([Xt], rng=[0, 1])[0]
    check_data(Xi, Xn, label="Inverse Compose rng 0-1")

    # Range 1-2
    Xt = mktr.transform([Xn], rng=[1, 2])[0]

    Xr = np.vstack([Xn[:3].reshape(1, -1), Xn[1:4].reshape(1, -1)])
    check_data(Xt, Xr, label="Compose rng 1-2")

    Xi = mktr.inverse_transform([Xt], rng=[1, 2])[0]
    check_data(Xi, Xn, label="Inverse Compose rng 1-2")
