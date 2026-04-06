import numpy as np
import torch

from dymad.numerics import central_diff, complex_step, torch_jacobian


def f_scalar(x):
    return x**2 + 3 * x + 2


def df_scalar(x):
    return 2 * x + 3


def f_vector_np(x):
    return np.array([x[0] ** 2 + x[1], np.sin(x[0]) + np.cos(x[1])])


def df_vector_np(x):
    return np.array([[2 * x[0], 1], [np.cos(x[0]), -np.sin(x[1])]])


def f_vector_torch(x):
    return torch.stack([x[0] ** 2 + x[1], torch.sin(x[0]) + torch.cos(x[1])])


def df_vector_torch(x):
    return torch.stack(
        [
            torch.tensor([2 * x[0], 1], dtype=x.dtype),
            torch.tensor([torch.cos(x[0]), -torch.sin(x[1])], dtype=x.dtype),
        ]
    )


def test_finite_diff():
    eps = 1e-10

    # Test 1: Scalar function, scalar input
    x = 1.0
    df_an = df_scalar(x)
    df_cs = complex_step(f_scalar, x)
    df_fd = central_diff(f_scalar, x)
    assert np.abs(df_cs - df_an) < eps, "Scalar func, cs"
    assert np.abs(df_fd - df_an) < eps * 2, "Scalar func, fd"

    # Test 2: Vector function, vector input
    x = np.array([1.0, 0.5])
    df_an = df_vector_np(x)
    df_cs = complex_step(f_vector_np, x)
    df_fd = central_diff(f_vector_np, x)
    assert np.all(np.abs(df_cs - df_an) < eps), "Vector func, full, cs"
    assert np.all(np.abs(df_fd - df_an) < eps * 2), "Vector func, full, fd"

    # Test 3: Directional derivative
    v = np.array([0.5, 0.5])
    df_an = df_vector_np(x) @ v.reshape(-1, 1)
    df_cs = complex_step(f_vector_np, x, v=v)
    df_fd = central_diff(f_vector_np, x, v=v)
    assert np.all(np.abs(df_cs - df_an) < eps), "Vector func, directional, cs"
    assert np.all(np.abs(df_fd - df_an) < eps), "Vector func, directional, fd"


def test_torch_jacobian():
    eps = 1e-14

    # Test 1: Scalar function, scalar input
    x = 1.0
    df_an = df_scalar(x)
    df_tj = torch_jacobian(f_scalar, x)
    assert np.abs(df_tj - df_an) < eps, "Scalar func"

    # Test 2: Vector function, vector input
    x = np.array([1.0, 0.5])
    df_an = df_vector_np(x)
    df_tj = torch_jacobian(f_vector_torch, x)
    assert np.all(np.abs(df_tj - df_an) < eps), "Vector func, full"

    # Test 3: Directional derivative
    v = np.array([0.5, 0.5])
    df_an = df_vector_np(x) @ v.reshape(-1, 1)
    df_tj = torch_jacobian(f_vector_torch, x, v=v)
    assert np.all(np.abs(df_tj - df_an) < eps), "Vector func, directional"
