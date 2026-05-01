import numpy as np
from scipy.signal import savgol_filter

from dymad.io import DataInterface
from dymad.numerics import denoise


def train_case(data, sample, path):
    x_data, t_data = sample
    config_path = path / "ker_model_auto.yaml"
    config_mod = {
        "data": {"path": data},
        "transform_x": [{"type": "scaler", "mode": "std"}, {"type": "delay", "delay": 1}],
    }
    di = DataInterface(config_path=config_path, config_mod=config_mod)

    Zdel = di.encode(x_data)
    x_reco = di.decode(Zdel)

    Z1 = di.encode(x_data, rng=[0, 1])
    Z2 = di.encode(Z1, rng=[1, 2])
    X1 = di.decode(Z2, rng=[1, 2])
    X2 = di.decode(X1, rng=[0, 1])

    assert np.allclose(x_data, x_reco), "full autoencoding"
    assert np.allclose(Z1, X1), "autoencoding step 1"
    assert np.allclose(Z2, Zdel), "autoencoding step 2"
    assert np.allclose(X2, x_reco), "autoencoding recover"


def test_di(kp_data, kp_test, env_setup):
    train_case(kp_data, kp_test, env_setup)


def test_di_encode_applies_denoise_per_trajectory_for_batched_inputs(kp_data, env_setup):
    x_batch = np.load(kp_data)["x"]
    config_path = env_setup / "ker_model_auto.yaml"
    config_mod = {
        "data": {"path": kp_data},
        "transform_x": [
            {
                "type": "denoise",
                "method": "savgol",
                "window_length": 5,
                "polyorder": 2,
            }
        ],
    }
    di = DataInterface(config_path=config_path, config_mod=config_mod)

    actual = di.encode(x_batch)
    expected = savgol_filter(x_batch, window_length=5, polyorder=2, axis=1)

    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(di.decode(actual), actual, atol=1e-6, rtol=1e-6)


def test_di_encode_supports_kernel_smoothing_per_trajectory(kp_data, env_setup):
    x_batch = np.load(kp_data)["x"]
    config_path = env_setup / "ker_model_auto.yaml"
    config_mod = {
        "data": {"path": kp_data},
        "transform_x": [
            {
                "type": "denoise",
                "method": "kernel_smoothing",
                "kernel": "gaussian",
                "anchor_count": 8,
                "bandwidth_multiplier": 2.0,
            }
        ],
    }
    di = DataInterface(config_path=config_path, config_mod=config_mod)

    actual = di.encode(x_batch)
    expected = np.stack(
        [
            denoise(
                trajectory,
                method="kernel_smoothing",
                kernel="gaussian",
                anchor_count=8,
                bandwidth_multiplier=2.0,
            )
            for trajectory in x_batch
        ],
        axis=0,
    )

    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(di.decode(actual), actual, atol=1e-6, rtol=1e-6)
