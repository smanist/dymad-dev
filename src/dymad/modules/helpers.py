from functools import partial
import torch
import torch.nn as nn
try:
    from torch_geometric.nn.conv import MessagePassing
    from torch_geometric.nn import ChebConv, GATConv, GCNConv, GraphConv, SAGEConv
except:
    MessagePassing = None
    ChebConv, GATConv, GCNConv, GraphConv, SAGEConv = None, None, None, None, None
from typing import Callable

#: Mapping of activation names to activation classes.
ACT_MAP = {
    "relu"     : nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
    "prelu"    : nn.PReLU,
    "tanh"     : nn.Tanh,
    "sigmoid"  : nn.Sigmoid,
    "gelu"     : nn.GELU,
    "silu"     : nn.SiLU,
    "swish"    : nn.SiLU,       # swish == SiLU in PyTorch
    "elu"      : nn.ELU,
    "selu"     : nn.SELU,
    "softplus" : nn.Softplus,
    "mish"     : nn.Mish,
    "none"     : nn.Identity,
}

#: Mapping of graph convolutional layer names to classes.
GCL_MAP = {
    "cheb"     : ChebConv,
    "gat"      : GATConv,
    "gcn"      : GCNConv,
    "gcnv"     : GraphConv,
    "sage"     : SAGEConv,
}

#: Mapping of weight initialization names to functions.
INIT_MAP_W = {
    "kaiming_uniform": nn.init.kaiming_uniform_,
    "kaiming_normal":  nn.init.kaiming_normal_,
    "xavier_uniform":  nn.init.xavier_uniform_,
    "xavier_normal":   nn.init.xavier_normal_,
    "orthogonal":      nn.init.orthogonal_,
    "normal":          nn.init.normal_,
    "trunc_normal":    nn.init.trunc_normal_,  # PyTorch ≥1.12
    "uniform":         nn.init.uniform_,
}

#: Mapping of bias initialization names to functions.
INIT_MAP_B = {
    # aliases -> torch.nn.init functions
    "zeros": nn.init.zeros_,
    "ones":  nn.init.ones_,
}

def _resolve_activation(spec, dtype, device) -> nn.Module:
    """
    Turn a user-supplied activation *specification* into an nn.Module.
    `spec` can be a string, an activation class, or a constructed module.
    """
    # case 1 ─ string
    if isinstance(spec, str):
        key = spec.lower()
        if key not in ACT_MAP:
            raise ValueError(f"Unknown activation string '{spec}'. "
                             f"Valid keys are {sorted(ACT_MAP.keys())}.")
        if key == "prelu":
            # dtype of the slope
            return partial(ACT_MAP[key], dtype=dtype, device=device)
        return ACT_MAP[key]

    # case 2 ─ activation *class* (subclass of nn.Module)
    if isinstance(spec, type) and issubclass(spec, nn.Module):
        return spec

    # case 3 ─ already-constructed module
    if isinstance(spec, nn.Module):
        return type(spec)

    raise TypeError("activation must be str, nn.Module subclass, "
                    f"or nn.Module instance, got {type(spec)}")

def _resolve_gcl(spec, opts) -> nn.Module:
    """
    Turn a user-supplied graph convolutional layer *specification* into an nn.Module.
    `spec` can be a string, a GCL class, or a constructed module.
    """
    # case 1 ─ string
    if isinstance(spec, str):
        key = spec.lower()
        if key not in GCL_MAP:
            raise ValueError(f"Unknown GCL string '{spec}'. "
                             f"Valid keys are {sorted(GCL_MAP.keys())}.")
        return lambda in_dim, out_dim: GCL_MAP[key](in_dim, out_dim, **opts)

    # case 2 ─ GCL *class* (subclass of MessagePassing)
    if isinstance(spec, type) and issubclass(spec, MessagePassing):
        return spec

    # case 3 ─ already-constructed module
    if isinstance(spec, MessagePassing):
        return type(spec)

    raise TypeError("GCL must be str, MessagePassing subclass, "
                    f"or MessagePassing instance, got {type(spec)}")

def _resolve_init(spec, map: str) -> Callable[[torch.Tensor, float], None]:
    """Turn <spec> (str | callable) into an init function."""
    if isinstance(spec, str):
        key = spec.lower()
        if key not in map:
            raise ValueError(f"Unknown init '{spec}'. Valid: {sorted(map)}")
        return map[key]
    if callable(spec):
        return spec
    raise TypeError("Init function must be str or callable")
