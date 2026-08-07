import jax
import jax.numpy as jnp
import pytest
import torch

from dymad.utils import JaxWrapper
from dymad.utils.wrapper import torch_to_jax

JAX_GPU_AVAILABLE = any(device.platform == "gpu" for device in jax.devices())


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


@pytest.mark.skipif(
    not torch.cuda.is_available() or not JAX_GPU_AVAILABLE,
    reason="PyTorch and JAX must both have CUDA available",
)
def test_jax_wrapper_preserves_cuda_device_and_gradients():
    def f_jax(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.tanh(x) + x**2

    x = torch.randn(8, 4, device="cuda", dtype=torch.float64, requires_grad=True)
    reference = x.detach().clone().requires_grad_(True)
    jax_x = torch_to_jax(x.detach())

    assert jax_x is not None
    assert jax_x.device.platform == "gpu"

    actual = JaxWrapper(f_jax, jit=True)(x)
    actual.sum().backward()
    expected = torch.tanh(reference) + reference**2
    expected.sum().backward()

    assert actual.device.type == "cuda"
    assert x.grad is not None
    assert torch.allclose(actual, expected, rtol=1e-6, atol=1e-7)
    assert torch.allclose(x.grad, reference.grad, rtol=1e-6, atol=1e-7)
