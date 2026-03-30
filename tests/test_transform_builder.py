from __future__ import annotations

import numpy as np
import torch
import pytest

from dymad.core import NDRTransformModuleAdapter, build_transform_module
from dymad.transform import make_transform


@pytest.mark.parametrize(
    ("config", "rtol"),
    [
        ({"type": "Isomap", "edim": 2, "Knn": 12, "inverse": "gmls", "order": 1, "Kphi": 4}, 1e-5),
        ({"type": "DiffMap", "edim": 2, "mode": "knn", "Knn": 12, "inverse": "pinv"}, 1e-4),
        ({"type": "DiffMapVB", "edim": 2, "mode": "knn", "Knn": 12, "inverse": "gmls", "order": 1, "Kphi": 4}, 1e-4),
    ],
)
def test_build_transform_module_wraps_ndr_transforms(config, rtol: float) -> None:
    tt = np.linspace(0.0, np.pi, 81)
    cc = np.cos(tt)
    ss = np.sin(tt)
    mixing = np.random.default_rng(7).random((2, 6))
    payload = (np.vstack([cc, ss]).T @ mixing).astype(np.float32)

    legacy = make_transform(config)
    legacy.fit([payload])
    legacy_out = legacy.transform([payload])[0]
    state = legacy.state_dict()

    module = build_transform_module(config, state)

    assert module.supports_gradients == "false"
    assert module.invertibility == "approximate"
    assert isinstance(module.transforms[0], NDRTransformModuleAdapter)

    actual = module.transform_batch([torch.as_tensor(payload)])[0].cpu().numpy()
    recovered = module.inverse_batch([torch.as_tensor(actual)])[0].cpu().numpy()

    np.testing.assert_allclose(actual, legacy_out, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(recovered, legacy.inverse_transform([legacy_out])[0], rtol=rtol, atol=rtol)
