from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable

import jax
import jax.numpy as jnp
import torch

from dymad.utils import JaxWrapper
from dymad.utils.wrapper import torch_to_jax


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the PyTorch-to-JAX autograd bridge.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    return parser.parse_args()


def _jax_dynamics(
    x: jax.Array,
    u: jax.Array,
    correction: jax.Array,
    params: jax.Array,
) -> jax.Array:
    velocity = x[..., 1]
    acceleration = -(9.81 / params[..., 0]) * (jnp.sin(x[..., 0]) + correction[..., 0] + u[..., 0])
    return jnp.stack((velocity, acceleration), axis=-1)


def _torch_dynamics(
    x: torch.Tensor,
    u: torch.Tensor,
    correction: torch.Tensor,
    params: torch.Tensor,
) -> torch.Tensor:
    velocity = x[..., 1]
    acceleration = -(9.81 / params[..., 0]) * (
        torch.sin(x[..., 0]) + correction[..., 0] + u[..., 0]
    )
    return torch.stack((velocity, acceleration), dim=-1)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure(
    step: Callable[[], None],
    *,
    device: torch.device,
    warmup: int,
    iters: int,
) -> dict[str, float]:
    for _ in range(warmup):
        step()
    _synchronize(device)

    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        step()
        _synchronize(device)
        samples.append(time.perf_counter() - start)
    return {
        "mean_ms": 1.0e3 * statistics.mean(samples),
        "median_ms": 1.0e3 * statistics.median(samples),
        "min_ms": 1.0e3 * min(samples),
        "max_ms": 1.0e3 * max(samples),
    }


def _describe(stats: dict[str, float]) -> str:
    return (
        f"mean={stats['mean_ms']:.3f}ms "
        f"median={stats['median_ms']:.3f}ms "
        f"min={stats['min_ms']:.3f}ms "
        f"max={stats['max_ms']:.3f}ms"
    )


def main() -> int:
    args = _parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.iters <= 0:
        raise ValueError("--iters must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access a CUDA device")

    device = torch.device(args.device)
    tensors = (
        torch.randn(args.batch_size, 2, device=device),
        torch.randn(args.batch_size, 1, device=device),
        torch.randn(args.batch_size, 1, device=device),
        torch.rand(args.batch_size, 1, device=device) + 0.5,
    )
    jax_layer = JaxWrapper(_jax_dynamics, jit=True)
    jax_input = torch_to_jax(tensors[0].detach())
    if jax_input is None:
        raise RuntimeError("Unable to convert the benchmark input to JAX")
    jax_input_device = jax_input.device

    def run_jax_wrapper() -> None:
        inputs = tuple(tensor.detach().requires_grad_(True) for tensor in tensors)
        jax_layer(*inputs).square().mean().backward()

    def run_torch() -> None:
        inputs = tuple(tensor.detach().requires_grad_(True) for tensor in tensors)
        _torch_dynamics(*inputs).square().mean().backward()

    print(
        f"torch_device={device} jax_input_device={jax_input_device} "
        f"batch_size={args.batch_size} "
        f"iters={args.iters} warmup={args.warmup}"
    )
    for name, step in (("jax_wrapper", run_jax_wrapper), ("torch", run_torch)):
        stats = _measure(step, device=device, warmup=args.warmup, iters=args.iters)
        print(f"{name:16s} {_describe(stats)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
