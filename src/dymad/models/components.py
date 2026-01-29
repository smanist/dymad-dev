import torch
import torch.nn as nn
from typing import Tuple

from dymad.io import DynData

# ------------------
# Encoder functions
# ------------------

def enc_iden(net: nn.Module, w: DynData) -> torch.Tensor:
    """Identity encoder function."""
    return w.x

def enc_smpl_auto(net: nn.Module, w: DynData) -> torch.Tensor:
    """Only encodes states."""
    return net(w.x)

def enc_smpl_ctrl(net: nn.Module, w: DynData) -> torch.Tensor:
    """Encodes states and controls."""
    return net(torch.cat([w.x, w.u], dim=-1))

def enc_raw_ctrl(net: nn.Module, w: DynData) -> torch.Tensor:
    """Let encoder handle states and controls."""
    return net(w.x, w.u)

def enc_graph_iden(net: nn.Module, w: DynData) -> torch.Tensor:
    """Identity encoder function for graph data."""
    return w.xg

def enc_graph_auto(net: nn.Module, w: DynData) -> torch.Tensor:
    """Using GNN in EncAuto."""
    return w.g(net(w.xg, w.ei, w.ew, w.ea))

def enc_graph_ctrl(net: nn.Module, w: DynData) -> torch.Tensor:
    """Using GNN in EncCtrl."""
    xu_cat = torch.cat([w.xg, w.ug], dim=-1)
    return w.g(net(xu_cat, w.ei, w.ew, w.ea))

def enc_node_auto(net: nn.Module, w: DynData) -> torch.Tensor:
    """Using EncAuto for each node of graph."""
    return w.G(net(w.xg))      # G is needed for unified data structure

def enc_node_ctrl(net: nn.Module, w: DynData) -> torch.Tensor:
    """Using EncCtrl for each node of graph."""
    xu_cat = torch.cat([w.xg, w.ug], dim=-1)
    return w.G(net(xu_cat))    # G is needed for unified data structure

def enc_node_raw_ctrl(net: nn.Module, w: DynData) -> torch.Tensor:
    """Using EncCtrl for each node of graph, letting encoder handle the concatenation."""
    return w.G(net(w.xg, w.ug))    # G is needed for unified data structure

#: Mapping of encoder names to encoder functions.
ENC_MAP = {
    "iden"       : enc_iden,
    "smpl_auto"  : enc_smpl_auto,
    "smpl_ctrl"  : enc_smpl_ctrl,
    "raw_auto"   : enc_smpl_auto,  # Effectively same as smpl_auto
    "raw_ctrl"   : enc_raw_ctrl,
    "graph_iden" : enc_graph_iden,
    "graph_auto" : enc_graph_auto,
    "graph_ctrl" : enc_graph_ctrl,
    "node_iden"  : enc_iden,       # Effectively same as regular iden
    "node_auto"  : enc_node_auto,
    "node_ctrl"  : enc_node_ctrl,
    "node_raw_auto" : enc_node_auto,  # Effectively same as node_auto
    "node_raw_ctrl" : enc_node_raw_ctrl
}


# ------------------
# Decoder functions
# ------------------
def dec_iden(net: nn.Module, z: torch.Tensor, w: DynData) -> torch.Tensor:
    """Identity decoder function."""
    return z

def dec_auto(net: nn.Module, z: torch.Tensor, w: DynData) -> torch.Tensor:
    """Generic decoder function."""
    return net(z)

def dec_graph(net: nn.Module, z: torch.Tensor, w: DynData) -> torch.Tensor:
    """Graph decoder function."""
    return net(z, w.ei, w.ew, w.ea)

def dec_node(net: nn.Module, z: torch.Tensor, w: DynData) -> torch.Tensor:
    """Node-wise decoder function."""
    return w.G(net(w.g(z)))    # G is needed for unified data structure

#: Mapping of decoder names to decoder functions.
DEC_MAP = {
    "iden"  : dec_iden,
    "auto"  : dec_auto,
    "graph" : dec_graph,
    "node"  : dec_node,
}


# ------------------
# Dynamics modules - features
# ------------------

def zu_cat_none(z: torch.Tensor, w: DynData) -> torch.Tensor:
    """No concatenation, just return z."""
    return z

def zu_cat_smpl(z: torch.Tensor, w: DynData) -> torch.Tensor:
    """Simple concatenation of z and u."""
    return torch.cat([z, w.u], dim=-1)

def zu_blin_no_const(z: torch.Tensor, w: DynData) -> torch.Tensor:
    """Compute bilinear features without constant term."""
    z_u = (z.unsqueeze(-1) * w.u.unsqueeze(-2)).reshape(*z.shape[:-1], -1)
    return torch.cat([z, z_u], dim=-1)

def zu_blin_with_const(z: torch.Tensor, w: DynData) -> torch.Tensor:
    """Compute bilinear features with constant term."""
    z_u = (z.unsqueeze(-1) * w.u.unsqueeze(-2)).reshape(*z.shape[:-1], -1)
    return torch.cat([z, z_u, w.u], dim=-1)

def zu_cat_smpl_graph(z: torch.Tensor, w: DynData) -> torch.Tensor:
    """Simple concatenation of z and u on graph."""
    return torch.cat([z, w.ug], dim=-1)

def zu_blin_no_const_graph(z: torch.Tensor, w: DynData) -> torch.Tensor:
    """Compute bilinear features without constant term for graph data."""
    u_reshaped = w.ug
    z_u = (z.unsqueeze(-1) * u_reshaped.unsqueeze(-2)).reshape(*z.shape[:-1], -1)
    return torch.cat([z, z_u], dim=-1)

def zu_blin_with_const_graph(z: torch.Tensor, w: DynData) -> torch.Tensor:
    """Compute bilinear features with constant term for graph data."""
    u_reshaped = w.ug
    z_u = (z.unsqueeze(-1) * u_reshaped.unsqueeze(-2)).reshape(*z.shape[:-1], -1)
    return torch.cat([z, z_u, u_reshaped], dim=-1)

#: Mapping of feature concatenation names to functions.
FZU_MAP = {
    "none"                  : zu_cat_none,
    "cat"                   : zu_cat_smpl,
    "blin_no_const"         : zu_blin_no_const,
    "blin_with_const"       : zu_blin_with_const,
    "graph_cat"             : zu_cat_smpl_graph,
    "graph_blin_no_const"   : zu_blin_no_const_graph,
    "graph_blin_with_const" : zu_blin_with_const_graph
}


# ------------------
# Dynamics modules - composers
# ------------------

def dyn_direct(net: nn.Module, s: torch.Tensor, z: torch.Tensor, w: DynData) -> torch.Tensor:
    """Processing without control inputs."""
    return net(s)

def dyn_skip(net: nn.Module, s: torch.Tensor, z: torch.Tensor, w: DynData) -> torch.Tensor:
    """Processing with skip connection."""
    return z + net(s)

def dyn_graph_direct(net: nn.Module, s: torch.Tensor, z: torch.Tensor, w: DynData) -> torch.Tensor:
    """Processing by GNN."""
    return net(w.g(s), w.ei, w.ew, w.ea)   # G is effectively applied in the net

def dyn_graph_skip(net: nn.Module, s: torch.Tensor, z: torch.Tensor, w: DynData) -> torch.Tensor:
    """Processing by GNN with skip connection."""
    return z + net(w.g(s), w.ei, w.ew, w.ea)   # G is effectively applied in the net

#: Mapping of dynamics composer names to functions.
DYN_MAP = {
    "direct"       : dyn_direct,
    "skip"         : dyn_skip,
    "graph_direct" : dyn_graph_direct,
    "graph_skip"   : dyn_graph_skip
}

# ------------------
# Dynamics modules - linear features
# ------------------

def linear_eval_smpl(mdl, w: DynData) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute linear evaluation, dz, and states, z, for the model."""
    z = mdl.encoder(w)
    z_dot = mdl.dynamics(z, w)
    return z_dot, z

def linear_features_smpl(mdl, w: DynData) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute linear features, f, and outputs, dz, for the model."""
    z = mdl.encoder(w)
    return mdl.features(z, w), z

def linear_eval_graph(mdl, w: DynData) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute linear evaluation, dz, and states, z, for the model."""
    z = mdl.encoder(w)
    z_dot = mdl.dynamics(z, w)
    return z_dot.permute(0, 2, 1, 3), z.permute(0, 2, 1, 3)

def linear_features_graph(mdl, w: DynData) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute linear features, f, and outputs, dz, for the model."""
    z = mdl.encoder(w)
    f = mdl.features(z, w)
    return f.permute(0, 2, 1, 3), z.permute(0, 2, 1, 3)

#: Mapping of linear evaluation and features functions.
LIN_MAP = {
    "smpl"  : (linear_eval_smpl, linear_features_smpl),
    "graph" : (linear_eval_graph, linear_features_graph)
}
