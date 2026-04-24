from __future__ import annotations

import numpy as np
import pytest
import torch

from dymad.core import (
    DenoisingTransform,
    DiffMapTransform,
    DiffMapVBTransform,
    IsomapTransform,
    build_transform_module,
)
from dymad.core.transform_builder import export_transform_state


@pytest.mark.parametrize(
    ("config", "module_type", "rtol"),
    [
        (
            {"type": "Isomap", "edim": 2, "Knn": 12, "inverse": "gmls", "order": 1, "Kphi": 4},
            IsomapTransform,
            1e-5,
        ),
        (
            {"type": "DiffMap", "edim": 2, "mode": "knn", "Knn": 12, "inverse": "pinv"},
            DiffMapTransform,
            1e-4,
        ),
        (
            {
                "type": "DiffMapVB",
                "edim": 2,
                "mode": "knn",
                "Knn": 12,
                "inverse": "gmls",
                "order": 1,
                "Kphi": 4,
            },
            DiffMapVBTransform,
            1e-4,
        ),
    ],
)
def test_build_transform_module_wraps_external_transforms(config, module_type, rtol: float) -> None:
    tt = np.linspace(0.0, np.pi, 81)
    cc = np.cos(tt)
    ss = np.sin(tt)
    mixing = np.random.default_rng(7).random((2, 6))
    payload = (np.vstack([cc, ss]).T @ mixing).astype(np.float32)

    module = build_transform_module(config)
    module.fit([torch.as_tensor(payload)])
    state = export_transform_state(module)

    reloaded = build_transform_module(config, state)

    assert reloaded.supports_gradients == "approximate"
    assert reloaded.invertibility == "approximate"
    assert isinstance(reloaded.transforms[0], module_type)

    actual = reloaded.transform_batch([torch.as_tensor(payload)])[0].cpu().numpy()
    recovered = reloaded.inverse_batch([torch.as_tensor(actual)])[0].cpu().numpy()

    expected = module.transform_batch([torch.as_tensor(payload)])[0].cpu().numpy()
    reference_inverse = module.inverse_batch([torch.as_tensor(expected)])[0].cpu().numpy()

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(recovered, reference_inverse, rtol=rtol, atol=rtol)


def test_build_transform_module_supports_denoise_config_and_state_reload() -> None:
    payload = np.array(
        [
            [0.0, 0.1],
            [0.5, 0.0],
            [1.0, -0.1],
            [0.5, 0.0],
            [0.0, 0.1],
        ],
        dtype=np.float32,
    )
    config = {"type": "Denoise", "method": "savgol", "window_length": 5, "polyorder": 2}

    module = build_transform_module(config)
    module.fit([torch.as_tensor(payload)])
    state = export_transform_state(module)
    reloaded = build_transform_module({"type": "denoise"}, state)

    assert isinstance(module.transforms[0], DenoisingTransform)
    assert isinstance(reloaded.transforms[0], DenoisingTransform)
    assert state["children"][0]["method"] == "savgol"
    assert state["children"][0]["kwargs"] == {"window_length": 5, "polyorder": 2}

    actual = reloaded.transform_batch([torch.as_tensor(payload)])[0]
    expected = module.transform_batch([torch.as_tensor(payload)])[0]
    torch.testing.assert_close(actual, expected)
