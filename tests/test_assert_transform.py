from __future__ import annotations

import numpy as np
import torch

from dymad.core import (
    AddOneTransform,
    ComposeTransform,
    DelayEmbeddingTransform,
    IdentityTransform,
    ScalerTransform,
    SVDTransform,
    build_transform_module,
)
from dymad.core.transform_builder import export_transform_state


def _fit(module, arrays):
    module.fit([torch.as_tensor(item, dtype=torch.float64) for item in arrays])
    return module


def check_data(out, ref, label=""):
    for _s, _t in zip(out, ref, strict=False):
        assert np.allclose(_s, _t), f"{label} failed: {_s} != {_t}"


Xs = [
    np.array([[1.0, 2.0], [1.1, 3.0], [1.2, 4.0], [1.3, 5.0], [1.4, 6.0], [1.5, 7.0]]),
    np.array([[2.2, 3.4], [2.3, 3.5], [2.4, 3.6], [2.5, 3.7]]),
    np.array([[1.0, 2.5], [1.2, 4.5], [1.4, 6.5], [1.6, 8.5]]),
]
Xn = np.array([[1.32, 2.4], [1.33, 3.5], [1.34, 4.6], [1.35, 5.7]])


def test_addone():
    addo = AddOneTransform()
    Xt = addo.transform(Xs)
    Xr = [np.concatenate([x, np.ones((len(x), 1))], axis=-1) for x in Xs]
    check_data(Xt, Xr, label="AddOne")
    Xi = addo.inverse_transform(Xt)
    check_data(Xi, Xs, label="Inverse AddOne")


def test_identity():
    iden = IdentityTransform()
    Xt = iden.transform(Xs)
    check_data(Xt, Xs, label="Identity")
    Xi = iden.inverse_transform(Xt)
    check_data(Xi, Xs, label="Inverse Identity")


def test_scaler():
    sclr = _fit(ScalerTransform(mode="01"), Xs)
    Xt = sclr.transform([Xn])[0]

    tmp = np.vstack(Xs)
    mx, mn = np.max(tmp, axis=0), np.min(tmp, axis=0)
    Xr = (Xn - mn) / (mx - mn)
    check_data([Xt], [Xr], label="Scalar 01")

    Xi = sclr.inverse_transform([Xt])[0]
    check_data([Xi], [Xn], label="Inverse Scalar 01")


def test_delay():
    dely = _fit(DelayEmbeddingTransform(delay=2), Xs)
    Xt = dely.transform([Xn])[0]

    Xr = np.vstack([Xn[:3].reshape(1, -1), Xn[1:4].reshape(1, -1)])
    check_data([Xt], [Xr], label="Delay")

    Xi = dely.inverse_transform([Xt])[0]
    check_data([Xi], [Xn], label="Inverse Delay")


def test_svd():
    svd = _fit(SVDTransform(order=2, ifcen=True), Xs)
    Xt = svd.transform([Xn])[0]

    tmp = np.vstack(Xs)
    avr = np.mean(tmp, axis=0)
    tmp = tmp - avr
    _, _, Vh = np.linalg.svd(tmp, full_matrices=False)
    Xr = (Xn - avr).dot(Vh[:2].T)
    check_data([Xt], [Xr], label="SVD")

    Xi = svd.inverse_transform([Xt])[0]
    check_data([Xi], [Xn], label="Inverse SVD")

    state = export_transform_state(svd)
    reld = build_transform_module({"type": "svd", "order": 2, "ifcen": True}, state)

    Xt = reld.transform([Xn])[0]
    check_data([Xt], [Xr], label="SVD reload")
    Xi = reld.inverse_transform([Xt])[0]
    check_data([Xi], [Xn], label="Inverse SVD reload")


def test_compose():
    config = [{"type": "scaler", "mode": "std"}, {"type": "delay", "delay": 1}]
    cmps = ComposeTransform([ScalerTransform(mode="std"), DelayEmbeddingTransform(delay=1)])
    _fit(cmps, Xs)
    Xt = cmps.transform([Xn])[0]

    tmp = np.vstack(Xs)
    avr, std = np.mean(tmp, axis=0), np.std(tmp, axis=0)
    tmp = (Xn - avr) / std
    Xr = np.vstack([tmp[:2].reshape(1, -1), tmp[1:3].reshape(1, -1), tmp[2:4].reshape(1, -1)])
    check_data([Xt], [Xr], label="Compose")

    Xi = cmps.inverse_transform([Xt])[0]
    check_data([Xi], [Xn], label="Inverse Compose")

    mktr = build_transform_module(config)
    _fit(mktr, Xs)
    Xt = mktr.transform([Xn])[0]
    check_data([Xt], [Xr], label="Compose build")

    Xi = mktr.inverse_transform([Xt])[0]
    check_data([Xi], [Xn], label="Inverse Compose build")

    state = export_transform_state(mktr)
    reld = build_transform_module(config, state)
    Xt = reld.transform([Xn])[0]
    check_data([Xt], [Xr], label="Compose reload")
    Xi = reld.inverse_transform([Xt])[0]
    check_data([Xi], [Xn], label="Inverse Compose reload")


def test_compose_rng():
    config = [{"type": "scaler", "mode": "01"}, {"type": "delay", "delay": 2}]
    mktr = build_transform_module(config)
    _fit(mktr, Xs)

    Xt = mktr.transform([Xn], rng=[0, 1])[0]
    tmp = np.vstack(Xs)
    mx, mn = np.max(tmp, axis=0), np.min(tmp, axis=0)
    Xr = (Xn - mn) / (mx - mn)
    check_data([Xt], [Xr], label="Compose rng 0-1")
    Xi = mktr.inverse_transform([Xt], rng=[0, 1])[0]
    check_data([Xi], [Xn], label="Inverse Compose rng 0-1")

    Xt = mktr.transform([Xn], rng=[1, 2])[0]
    Xr = np.vstack([Xn[:3].reshape(1, -1), Xn[1:4].reshape(1, -1)])
    check_data([Xt], [Xr], label="Compose rng 1-2")
    Xi = mktr.inverse_transform([Xt], rng=[1, 2])[0]
    check_data([Xi], [Xn], label="Inverse Compose rng 1-2")


def test_negative_order_svd_uses_matrix_aspect_ratio() -> None:
    rng = np.random.default_rng(3)
    payload = rng.normal(size=(100, 5)).astype(np.float64)

    module = build_transform_module({"type": "svd", "order": -1.0})
    module.fit([torch.as_tensor(payload, dtype=torch.float64)])

    singular_values = np.linalg.svd(payload, full_matrices=False, compute_uv=False)
    beta = min(payload.shape) / max(payload.shape)
    omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
    expected_rank = np.argmax(singular_values < omega * np.median(singular_values))
    expected_rank = max(1, int(expected_rank))

    assert module.transforms[0].output_dim == expected_rank
