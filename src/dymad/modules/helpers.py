from collections.abc import Callable
from functools import partial
from typing import Any, Literal, cast

import torch
import torch.nn as nn

try:
    from torch_geometric.nn import ChebConv, GATConv, GCNConv, GraphConv, SAGEConv
    from torch_geometric.nn.conv import MessagePassing
except ImportError:
    MessagePassing = None
    ChebConv, GATConv, GCNConv, GraphConv, SAGEConv = None, None, None, None, None

ActivationFactory = Callable[[], nn.Module]
InitFn = Callable[..., Any]


def _swap_parameter_storage(
    param: nn.Parameter, tensor: torch.Tensor, requires_grad: bool | None = None
) -> None:
    if requires_grad is not None and param.is_leaf:
        param.requires_grad_(False)
    cast(Any, param).set_(tensor.to(param))
    if requires_grad is not None and param.is_leaf:
        param.requires_grad_(requires_grad)


def _gain_nonlinearity(act_name: str) -> Literal["relu", "tanh", "sigmoid"]:
    if act_name == "tanh":
        return "tanh"
    if act_name == "sigmoid":
        return "sigmoid"
    return "relu"


#: Mapping of activation names to activation classes.
ACT_MAP: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
    "prelu": nn.PReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "swish": nn.SiLU,  # swish == SiLU in PyTorch
    "elu": nn.ELU,
    "selu": nn.SELU,
    "softplus": nn.Softplus,
    "mish": nn.Mish,
    "none": nn.Identity,
}

#: Mapping of graph convolutional layer names to classes.
GCL_MAP: dict[str, type[nn.Module] | None] = {
    "cheb": ChebConv,
    "gat": GATConv,
    "gcn": GCNConv,
    "gcnv": GraphConv,
    "sage": SAGEConv,
}

#: Mapping of weight initialization names to functions.
INIT_MAP_W: dict[str, InitFn] = {
    "kaiming_uniform": nn.init.kaiming_uniform_,
    "kaiming_normal": nn.init.kaiming_normal_,
    "xavier_uniform": nn.init.xavier_uniform_,
    "xavier_normal": nn.init.xavier_normal_,
    "orthogonal": nn.init.orthogonal_,
    "normal": nn.init.normal_,
    "trunc_normal": nn.init.trunc_normal_,  # PyTorch ≥1.12
    "uniform": nn.init.uniform_,
}

#: Mapping of bias initialization names to functions.
INIT_MAP_B: dict[str, InitFn] = {
    # aliases -> torch.nn.init functions
    "zeros": nn.init.zeros_,
    "ones": nn.init.ones_,
}


def _resolve_activation(spec, dtype, device) -> ActivationFactory:
    """
    Turn a user-supplied activation *specification* into an nn.Module.
    `spec` can be a string, an activation class, or a constructed module.
    """
    # case 1 ─ string
    if isinstance(spec, str):
        key = spec.lower()
        if key not in ACT_MAP:
            raise ValueError(
                f"Unknown activation string '{spec}'. Valid keys are {sorted(ACT_MAP.keys())}."
            )
        if key == "prelu":
            # dtype of the slope
            return cast(ActivationFactory, partial(ACT_MAP[key], dtype=dtype, device=device))
        return cast(ActivationFactory, ACT_MAP[key])

    # case 2 ─ activation *class* (subclass of nn.Module)
    if isinstance(spec, type) and issubclass(spec, nn.Module):
        return cast(ActivationFactory, spec)

    # case 3 ─ already-constructed module
    if isinstance(spec, nn.Module):
        return cast(ActivationFactory, type(spec))

    raise TypeError(
        f"activation must be str, nn.Module subclass, or nn.Module instance, got {type(spec)}"
    )


def _resolve_gcl(spec, opts) -> Callable[[int, int], nn.Module]:
    """
    Turn a user-supplied graph convolutional layer *specification* into an nn.Module.
    `spec` can be a string, a GCL class, or a constructed module.
    """
    # case 1 ─ string
    if isinstance(spec, str):
        key = spec.lower()
        if key not in GCL_MAP:
            raise ValueError(
                f"Unknown GCL string '{spec}'. Valid keys are {sorted(GCL_MAP.keys())}."
            )
        gcl_cls = GCL_MAP[key]
        if gcl_cls is None:
            raise ImportError("torch_geometric is required to build graph convolution layers.")
        return lambda in_dim, out_dim: gcl_cls(in_dim, out_dim, **opts)

    # case 2 ─ GCL *class* (subclass of MessagePassing)
    if MessagePassing is not None and isinstance(spec, type) and issubclass(spec, MessagePassing):
        return cast(Callable[[int, int], nn.Module], spec)

    # case 3 ─ already-constructed module
    if MessagePassing is not None and isinstance(spec, MessagePassing):
        return cast(Callable[[int, int], nn.Module], type(spec))

    raise TypeError(
        f"GCL must be str, MessagePassing subclass, or MessagePassing instance, got {type(spec)}"
    )


def _resolve_init(spec, map: dict[str, InitFn]) -> InitFn:
    """Turn <spec> (str | callable) into an init function."""
    if isinstance(spec, str):
        key = spec.lower()
        if key not in map:
            raise ValueError(f"Unknown init '{spec}'. Valid: {sorted(map)}")
        return map[key]
    if callable(spec):
        return cast(InitFn, spec)
    raise TypeError("Init function must be str or callable")
