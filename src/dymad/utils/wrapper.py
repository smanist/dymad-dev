from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import Any, cast

import jax
import torch
from jax import dlpack as jax_dlpack
from torch.autograd import Function
from torch.utils import dlpack as torch_dlpack


# ---- Torch <-> JAX via DLPack ----
def torch_to_jax(t: torch.Tensor | None) -> jax.Array | None:
    if t is None:
        return None
    tensor = cast(torch.Tensor, t)
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return jax_dlpack.from_dlpack(tensor.detach())


def jax_to_torch(
    a: jax.Array,
    device: torch.device | None,
    dtype: torch.dtype | None,
) -> torch.Tensor:
    t = torch_dlpack.from_dlpack(a)
    if device is None and dtype is None:
        return t
    return t.to(device=device, dtype=dtype)


# -------------------- (Optional) tiny jit cache by id(f) --------------------
@lru_cache(maxsize=128)
def _jit_by_identity(fid: int, static: bool) -> Callable[..., Any]:
    # `fid` is just a cache key; we stash the real callable in a side dict
    f = _FUNC_REG[fid]
    return jax.jit(f) if static else f


_FUNC_REG: dict[int, Callable[..., Any]] = {}


def _get_jitted(f: Callable[..., Any], jit: bool) -> Callable[..., Any]:
    if not jit:
        return f
    fid = id(f)
    _FUNC_REG[fid] = f
    return _jit_by_identity(fid, True)


# -------------------- The interface --------------------
class JaxMultiInFn(Function):
    @staticmethod
    def forward(ctx, f_jax: Callable[..., Any], jit_flag: bool, *x_torch: torch.Tensor):
        """
        Accepts N torch tensors, all requiring grad.
        Returns either a single tensor or a tuple of tensors (matching f_jax).
        """
        # Save input metadata for dtype/device restoration
        in_devices = [None if t is None else t.device for t in x_torch]
        in_dtypes = [None if t is None else t.dtype for t in x_torch]

        # Torch -> JAX (zero-copy when possible)
        xs_jax = [torch_to_jax(t) for t in x_torch]

        # (Optionally) JIT the callable with a small cache by identity
        f_used = _get_jitted(f_jax, jit_flag)

        # y, pullback
        y_jax, pullback = jax.vjp(f_used, *xs_jax)
        ctx.pullback = pullback

        # Save input dtypes/devices for backward mapping
        ctx.in_devices = in_devices
        ctx.in_dtypes = in_dtypes

        out_device = next((d for d in in_devices if d is not None), None)
        out_dtype = next((d for d in in_dtypes if d is not None), None)

        ctx.out_is_tuple = isinstance(y_jax, tuple)

        # Remember output structure
        if ctx.out_is_tuple:
            return tuple(jax_to_torch(cast(jax.Array, y), out_device, out_dtype) for y in y_jax)
        return jax_to_torch(cast(jax.Array, y_jax), out_device, out_dtype)

    @staticmethod
    def backward(ctx, *grad_y_torch: torch.Tensor):
        """
        Receives one grad tensor per forward output (or one if single output).
        Returns one grad tensor per forward input (same order as inputs).
        """
        # Build cotangent pytree matching the JAX output structure
        if ctx.out_is_tuple:
            gy_jax = tuple(torch_to_jax(g) if g is not None else None for g in grad_y_torch)
        else:
            # grad_y_torch is a 1-tuple here
            (g,) = grad_y_torch
            gy_jax = torch_to_jax(g) if g is not None else None

        # VJP: returns tuple of cotangents matching inputs
        gx_jax_tuple = ctx.pullback(gy_jax)  # length == number of inputs

        # Map JAX grads back to Torch per-input device/dtype
        grads = []
        for gx_jax, dev, dt in zip(gx_jax_tuple, ctx.in_devices, ctx.in_dtypes, strict=False):
            if gx_jax is None:
                grads.append(None)
            else:
                grads.append(jax_to_torch(gx_jax, dev, dt))
        return (None, None, *grads)


# -------------------- Convenience nn.Module wrapper --------------------
class JaxWrapper(torch.nn.Module):
    def __init__(self, f_jax: Callable[..., Any], jit: bool = True):
        super().__init__()
        self.f_jax = f_jax
        self.jit = jit

    def forward(self, *xs: torch.Tensor):
        return JaxMultiInFn.apply(self.f_jax, self.jit, *xs)
