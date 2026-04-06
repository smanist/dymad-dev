import jax.numpy as jnp
import torch

from dymad.utils import JaxWrapper


def test_jax_wrapper():
    # JAX function
    def f_jax(*xs: jnp.ndarray) -> jnp.ndarray | tuple[jnp.ndarray, ...]:
        x, w = xs
        y1 = jnp.tanh(x @ w)
        y2 = jnp.sum(x**2, axis=-1)
        return y1, y2

    jax_layer = JaxWrapper(f_jax, jit=True)

    # Torch reference
    class TorchMultiInLayer(torch.nn.Module):
        def forward(self, *xs: torch.Tensor):
            x, w = xs
            y1 = torch.tanh(x @ w)
            y2 = torch.sum(x**2, axis=-1)
            return y1, y2

    tor_layer = TorchMultiInLayer()

    # Run two functions
    x = torch.randn(8, 16, requires_grad=True, dtype=torch.float64)
    w = torch.randn(16, 8, requires_grad=True, dtype=torch.float64)
    X = x.clone().detach().requires_grad_(True)
    W = w.clone().detach().requires_grad_(True)

    y1, y2 = jax_layer(x, w)
    loss = y1.pow(2).mean() + y2.mean()
    loss.backward()

    y1, y2 = tor_layer(X, W)
    loss = y1.pow(2).mean() + y2.mean()
    loss.backward()

    # Compare
    err = torch.linalg.norm(x.grad - X.grad) / torch.linalg.norm(x.grad)
    assert err.item() < 1e-7, "X grad"
    err = torch.linalg.norm(w.grad - W.grad) / torch.linalg.norm(w.grad)
    assert err.item() < 5e-6, "W grad"
