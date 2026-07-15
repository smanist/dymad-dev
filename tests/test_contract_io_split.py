import numpy as np

from dymad.core import ScalerTransform
from dymad.io import Split


def test_split_defaults_to_identity_transforms_and_preserves_raw_arrays() -> None:
    x_train = np.array([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]])
    y_train = np.array([[2.0], [4.0], [8.0]])
    x_val = np.array([[7.0, 14.0]])
    y_val = np.array([[10.0]])
    x_test = np.array([[9.0, 18.0]])
    y_test = np.array([[12.0]])

    split = Split.from_arrays(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
    )

    assert np.array_equal(split.x_train_raw, x_train)
    assert isinstance(split.x_transform, ScalerTransform)
    assert isinstance(split.y_transform, ScalerTransform)
    assert split.x_transform.mode == "none"
    assert split.y_transform.mode == "none"
    assert np.array_equal(split.x_train, x_train)
    assert np.array_equal(split.y_train, y_train)
    assert np.allclose(split.inverse_y(split.y_test), y_test)


def test_split_std_shortcut_fits_train_only_scalers() -> None:
    x_train = np.array([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]])
    y_train = np.array([[2.0], [4.0], [8.0]])
    x_val = np.array([[7.0, 14.0]])
    y_val = np.array([[10.0]])
    x_test = np.array([[9.0, 18.0]])
    y_test = np.array([[12.0]])

    split = Split.from_arrays(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        x_transform="std",
        y_transform="std",
    )

    assert isinstance(split.x_transform, ScalerTransform)
    assert isinstance(split.y_transform, ScalerTransform)
    assert split.x_transform.mode == "std"
    assert split.y_transform.mode == "std"
    assert np.allclose(split.x_train.mean(axis=0), 0.0)
    assert np.allclose(split.x_train.std(axis=0), 1.0)
    assert np.allclose(split.inverse_y(split.y_test), y_test)
