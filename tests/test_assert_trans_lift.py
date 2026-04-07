from __future__ import annotations

import numpy as np
import torch

from dymad.core import build_transform_module
from dymad.core.transform_builder import export_transform_state


def _fit(module, arrays):
    module.fit([torch.as_tensor(item, dtype=torch.float64) for item in arrays])
    return module


def _check_reload(config, arrays, expected=None):
    module = _fit(build_transform_module(config), arrays)
    transformed = module.transform(arrays)
    if expected is not None:
        for actual, target in zip(transformed, expected, strict=False):
            assert np.allclose(actual, target)
    recovered = module.inverse_transform(transformed)
    for actual, target in zip(recovered, arrays, strict=False):
        assert np.allclose(actual, target)

    reloaded = build_transform_module(config, export_transform_state(module))
    transformed_reload = reloaded.transform(arrays)
    recovered_reload = reloaded.inverse_transform(transformed_reload)
    for actual, target in zip(recovered_reload, arrays, strict=False):
        assert np.allclose(actual, target)


def test_poly():
    xs = np.array([[1.0, 2.0, -0.1], [1.1, 3.0, -0.2], [1.2, 4.0, -0.3], [1.3, 5.0, -0.4]])
    ks = [3, 2, 4]

    xp = []
    x1, x2, x3 = xs.T
    for k1 in range(ks[0]):
        for k2 in range(ks[1]):
            for k3 in range(ks[2]):
                xp.append((x1**k1) * (x2**k2) * (x3**k3))
    xp = np.vstack(xp).T

    _check_reload({"type": "lift", "fobs": "poly", "Ks": ks}, [xs, xs], [xp, xp])


def test_mixed_mfp():
    xs = np.array(
        [
            [1.0, 0.4, -0.1, 2.0],
            [1.1, 0.3, -0.2, 2.1],
            [1.2, -0.2, -0.3, 2.2],
            [1.3, -0.1, -0.4, 2.3],
        ]
    )
    ks = [5, 3, 2, 4]

    xp = []
    x1, x2, x3, x4 = xs.T
    radius = np.sqrt(x2**2 + x4**2)
    theta = np.arctan2(x2, x4)
    p1 = [np.ones_like(x3), np.cos(x3), np.sin(x3), np.cos(2 * x3), np.sin(2 * x3)]
    p2 = [
        np.ones_like(theta),
        np.cos(theta),
        np.sin(theta),
        np.cos(2 * theta),
        np.sin(2 * theta),
        np.cos(3 * theta),
        np.sin(3 * theta),
    ]
    for k1 in range(ks[0]):
        for k2 in range(2 * ks[1] + 1):
            for k3 in range(2 * ks[2] + 1):
                for k4 in range(ks[3]):
                    xp.append((x1**k1) * p2[k2] * p1[k3] * (radius**k4))
    xp = np.vstack(xp).T

    opts = [(0, "m", 5), (2, "f", 2), ([3, 1], "p", [4, 3])]
    _check_reload({"type": "lift", "fobs": "mixed", "opts": opts}, [xs, xs], [xp, xp])


def test_custom_finv():
    def fobs(x, a=1.0):
        return np.vstack([x[:, 0], np.exp(a * x[:, 1])]).T

    def finv(z, a=1.0):
        return np.vstack([z[:, 0], np.log(z[:, 1]) / a]).T

    xs = np.array([[1.0, 0.4], [1.1, 0.3], [1.2, 0.2], [1.3, 0.1]])
    xp = np.array(
        [[1.0, np.exp(0.4)], [1.1, np.exp(0.3)], [1.2, np.exp(0.2)], [1.3, np.exp(0.1)]]
    )

    _check_reload({"type": "lift", "fobs": fobs, "finv": finv, "a": 1.0}, [xs, xs], [xp, xp])


def test_custom_pinv():
    def fobs(x):
        return np.vstack([x[:, 0], x[:, 0] + x[:, 1]]).T

    xs = np.array([[1.0, 0.4], [1.1, 0.3], [1.2, 0.2], [1.3, 0.1]])
    xp = np.array([[1.0, 1.4], [1.1, 1.4], [1.2, 1.4], [1.3, 1.4]])

    _check_reload({"type": "lift", "fobs": fobs}, [xs, xs], [xp, xp])
