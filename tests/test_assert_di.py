import numpy as np

from dymad.io import DataInterface


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
