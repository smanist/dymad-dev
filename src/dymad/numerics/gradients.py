from typing import Any, cast

import numpy as np
import torch
import torch.func as func


def complex_step(f, x, h=1e-20, v=None):
    """Complex step differentiation.

    Args:
        f: function handle that accepts and returns numpy arrays.
        x: input array.
        h: step size.
        v: directions for directional derivative. If None, return full Jacobian.

    Returns:
        df: derivative of f at x, possibly directional.
    """
    # Process x
    x = np.asarray(x).squeeze()
    if np.iscomplexobj(x):
        raise ValueError("Input x should be real.")
    n = x.size
    # Process v
    if v is None:
        v = np.eye(n)
    else:
        v = np.asarray(v)
        if v.ndim == 1:
            v = v.reshape(1, -1)
        assert v.shape[1] == n, "v should have shape (m,n) where n is the size of x."
    # Complex step
    J = []
    for _v in v:
        J.append(np.imag(f(x + 1j * h * _v)) / h)
    return np.array(J).T


def central_diff(f, x, h=1e-6, v=None):
    """Central difference.

    Args:
        f: function handle that accepts and returns numpy arrays.
        x: input array.
        h: step size.
        v: directions for directional derivative. If None, return full Jacobian.

    Returns:
        df: derivative of f at x, possibly directional.
    """
    # Process x
    x = np.asarray(x).squeeze()
    if np.iscomplexobj(x):
        raise ValueError("Input x should be real.")
    n = x.size
    # Process v
    if v is None:
        v = np.eye(n)
    else:
        v = np.asarray(v)
        if v.ndim == 1:
            v = v.reshape(1, -1)
        assert v.shape[1] == n, "v should have shape (m,n) where n is the size of x."
    # Central difference
    J = []
    for _v in v:
        J.append((f(x + h * _v) - f(x - h * _v)) / (2 * h))
    return np.array(J).T


def torch_jacobian(f, x, v=None, dtype=torch.float64):
    """Jacobian using torch.func.jacobian.

    Args:
        f: function handle that accepts and returns torch tensors.
        x: input array.
        v: directions for directional derivative. If None, return full Jacobian.

    Returns:
        df: derivative of f at x, possibly directional.
    """
    x = torch.tensor(x, dtype=dtype)
    n = x.numel()

    if v is None:
        # Full Jacobian
        jac = func.jacrev(f)(x)
        jac = jac.reshape(-1, n)  # shape (output_dim, input_dim)
    else:
        # Directional derivative
        v = torch.tensor(v, dtype=dtype)
        if v.ndim == 1:
            v = v.reshape(1, -1)
        assert v.shape[1] == n, "v should have shape (m,n) where n is the size of x."
        jac = []
        for direction in v:
            _, jvp = cast(tuple[Any, torch.Tensor], func.jvp(f, (x,), (direction,)))
            jac.append(jvp)
        jac = torch.stack(jac, dim=1)  # shape (output_dim, num_directions)

    return jac.detach().numpy()
