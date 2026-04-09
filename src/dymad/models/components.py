import torch
import torch.nn as nn

from dymad.core.model_context import GraphModelContext
from dymad.core.runtime import RaggedGraphRuntime, UniformGraphRuntime
from dymad.models.runtime_view import (
    ComponentInputPayload,
    ComponentInputView,
    build_component_input_view,
)

# ------------------
# Encoder functions
# ------------------


def _require_control_from_view(view: ComponentInputView) -> torch.Tensor:
    control = view.control
    if control is None:
        raise ValueError("Control input is required for this component.")
    return control


def _require_control(w: ComponentInputPayload) -> torch.Tensor:
    return _require_control_from_view(build_component_input_view(w))


def _require_graph_control_from_view(view: ComponentInputView) -> torch.Tensor:
    control = view.graph_control
    if control is None:
        raise ValueError("Graph control input is required for this component.")
    return control


def _require_graph_control(w: ComponentInputPayload) -> torch.Tensor:
    return _require_graph_control_from_view(build_component_input_view(w))


def _require_graph_sequence_runtime(
    w: ComponentInputPayload,
) -> UniformGraphRuntime | RaggedGraphRuntime:
    if isinstance(w, GraphModelContext):
        return w.to_runtime()
    view = build_component_input_view(w)
    runtime = view.runtime
    if isinstance(runtime, (UniformGraphRuntime, RaggedGraphRuntime)):
        return runtime
    raise TypeError("Graph sequence runtime is required for graph linear evaluation.")


def enc_iden(net: nn.Module, w: ComponentInputPayload) -> torch.Tensor:
    """Identity encoder function."""
    return build_component_input_view(w).state


def enc_smpl_auto(net: nn.Module, w: ComponentInputPayload) -> torch.Tensor:
    """Only encodes states."""
    return net(build_component_input_view(w).state)


def enc_smpl_ctrl(net: nn.Module, w: ComponentInputPayload) -> torch.Tensor:
    """Encodes states and controls."""
    view = build_component_input_view(w)
    return net(torch.cat([view.state, _require_control_from_view(view)], dim=-1))


def enc_raw_ctrl(net: nn.Module, w: ComponentInputPayload) -> torch.Tensor:
    """Let encoder handle states and controls."""
    view = build_component_input_view(w)
    return net(view.state, view.control)


def enc_graph_iden(net: nn.Module, w: ComponentInputPayload) -> torch.Tensor:
    """Identity encoder function for graph data."""
    return build_component_input_view(w).graph_state


def enc_graph_auto(net: nn.Module, w: ComponentInputPayload) -> torch.Tensor:
    """Using GNN in EncAuto."""
    view = build_component_input_view(w)
    return view.unflatten_nodes(
        net(view.graph_state, view.edge_index, view.edge_weight, view.edge_attr)
    )


def enc_graph_ctrl(net: nn.Module, w: ComponentInputPayload) -> torch.Tensor:
    """Using GNN in EncCtrl."""
    view = build_component_input_view(w)
    xu_cat = torch.cat([view.graph_state, _require_graph_control_from_view(view)], dim=-1)
    return view.unflatten_nodes(net(xu_cat, view.edge_index, view.edge_weight, view.edge_attr))


def enc_node_auto(net: nn.Module, w: ComponentInputPayload) -> torch.Tensor:
    """Using EncAuto for each node of graph."""
    view = build_component_input_view(w)
    return view.flatten_nodes(net(view.graph_state))


def enc_node_ctrl(net: nn.Module, w: ComponentInputPayload) -> torch.Tensor:
    """Using EncCtrl for each node of graph."""
    view = build_component_input_view(w)
    xu_cat = torch.cat([view.graph_state, _require_graph_control_from_view(view)], dim=-1)
    return view.flatten_nodes(net(xu_cat))


def enc_node_raw_ctrl(net: nn.Module, w: ComponentInputPayload) -> torch.Tensor:
    """Using EncCtrl for each node of graph, letting encoder handle the concatenation."""
    view = build_component_input_view(w)
    return view.flatten_nodes(net(view.graph_state, _require_graph_control_from_view(view)))


#: Mapping of encoder names to encoder functions.
ENC_MAP = {
    "iden": enc_iden,
    "smpl_auto": enc_smpl_auto,
    "smpl_ctrl": enc_smpl_ctrl,
    "raw_auto": enc_smpl_auto,  # Effectively same as smpl_auto
    "raw_ctrl": enc_raw_ctrl,
    "graph_iden": enc_graph_iden,
    "graph_auto": enc_graph_auto,
    "graph_ctrl": enc_graph_ctrl,
    "node_iden": enc_iden,  # Effectively same as regular iden
    "node_auto": enc_node_auto,
    "node_ctrl": enc_node_ctrl,
    "node_raw_auto": enc_node_auto,  # Effectively same as node_auto
    "node_raw_ctrl": enc_node_raw_ctrl,
}


# ------------------
# Decoder functions
# ------------------
def dec_iden(net: nn.Module, z: torch.Tensor, w: ComponentInputPayload) -> torch.Tensor:
    """Identity decoder function."""
    return z


def dec_auto(net: nn.Module, z: torch.Tensor, w: ComponentInputPayload) -> torch.Tensor:
    """Generic decoder function."""
    return net(z)


def dec_graph(net: nn.Module, z: torch.Tensor, w: ComponentInputPayload) -> torch.Tensor:
    """Graph decoder function."""
    view = build_component_input_view(w)
    return net(z, view.edge_index, view.edge_weight, view.edge_attr)


def dec_node(net: nn.Module, z: torch.Tensor, w: ComponentInputPayload) -> torch.Tensor:
    """Node-wise decoder function."""
    view = build_component_input_view(w)
    return view.flatten_nodes(net(view.unflatten_nodes(z)))


#: Mapping of decoder names to decoder functions.
DEC_MAP = {
    "iden": dec_iden,
    "auto": dec_auto,
    "graph": dec_graph,
    "node": dec_node,
}


# ------------------
# Dynamics modules - features
# ------------------


def zu_cat_none(z: torch.Tensor, w: ComponentInputPayload) -> torch.Tensor:
    """No concatenation, just return z."""
    return z


def zu_cat_smpl(z: torch.Tensor, w: ComponentInputPayload) -> torch.Tensor:
    """Simple concatenation of z and u."""
    return torch.cat([z, _require_control(w)], dim=-1)


def zu_blin_no_const(z: torch.Tensor, w: ComponentInputPayload) -> torch.Tensor:
    """Compute bilinear features without constant term."""
    control = _require_control(w)
    z_u = (z.unsqueeze(-1) * control.unsqueeze(-2)).reshape(*z.shape[:-1], -1)
    return torch.cat([z, z_u], dim=-1)


def zu_blin_with_const(z: torch.Tensor, w: ComponentInputPayload) -> torch.Tensor:
    """Compute bilinear features with constant term."""
    control = _require_control(w)
    z_u = (z.unsqueeze(-1) * control.unsqueeze(-2)).reshape(*z.shape[:-1], -1)
    return torch.cat([z, z_u, control], dim=-1)


def zu_cat_smpl_graph(z: torch.Tensor, w: ComponentInputPayload) -> torch.Tensor:
    """Simple concatenation of z and u on graph."""
    return torch.cat([z, _require_graph_control(w)], dim=-1)


def zu_blin_no_const_graph(z: torch.Tensor, w: ComponentInputPayload) -> torch.Tensor:
    """Compute bilinear features without constant term for graph data."""
    u_reshaped = _require_graph_control(w)
    z_u = (z.unsqueeze(-1) * u_reshaped.unsqueeze(-2)).reshape(*z.shape[:-1], -1)
    return torch.cat([z, z_u], dim=-1)


def zu_blin_with_const_graph(z: torch.Tensor, w: ComponentInputPayload) -> torch.Tensor:
    """Compute bilinear features with constant term for graph data."""
    u_reshaped = _require_graph_control(w)
    z_u = (z.unsqueeze(-1) * u_reshaped.unsqueeze(-2)).reshape(*z.shape[:-1], -1)
    return torch.cat([z, z_u, u_reshaped], dim=-1)


#: Mapping of feature concatenation names to functions.
FZU_MAP = {
    "none": zu_cat_none,
    "cat": zu_cat_smpl,
    "blin_no_const": zu_blin_no_const,
    "blin_with_const": zu_blin_with_const,
    "graph_cat": zu_cat_smpl_graph,
    "graph_blin_no_const": zu_blin_no_const_graph,
    "graph_blin_with_const": zu_blin_with_const_graph,
}


# ------------------
# Dynamics modules - composers
# ------------------


def dyn_direct(
    net: nn.Module, s: torch.Tensor, z: torch.Tensor, w: ComponentInputPayload
) -> torch.Tensor:
    """Processing without control inputs."""
    return net(s)


def dyn_skip(
    net: nn.Module, s: torch.Tensor, z: torch.Tensor, w: ComponentInputPayload
) -> torch.Tensor:
    """Processing with skip connection."""
    return z + net(s)


def dyn_graph_direct(
    net: nn.Module, s: torch.Tensor, z: torch.Tensor, w: ComponentInputPayload
) -> torch.Tensor:
    """Processing by GNN."""
    view = build_component_input_view(w)
    return net(view.unflatten_nodes(s), view.edge_index, view.edge_weight, view.edge_attr)


def dyn_graph_skip(
    net: nn.Module, s: torch.Tensor, z: torch.Tensor, w: ComponentInputPayload
) -> torch.Tensor:
    """Processing by GNN with skip connection."""
    view = build_component_input_view(w)
    return z + net(view.unflatten_nodes(s), view.edge_index, view.edge_weight, view.edge_attr)


#: Mapping of dynamics composer names to functions.
DYN_MAP = {
    "direct": dyn_direct,
    "skip": dyn_skip,
    "graph_direct": dyn_graph_direct,
    "graph_skip": dyn_graph_skip,
}

# ------------------
# Dynamics modules - linear features
# ------------------


def linear_eval_smpl(mdl, w: ComponentInputPayload) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute linear evaluation, dz, and states, z, for the model."""
    z = mdl.encoder(w)
    z_dot = mdl.dynamics(z, w)
    return z_dot, z


def linear_features_smpl(mdl, w: ComponentInputPayload) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute linear features, f, and outputs, dz, for the model."""
    z = mdl.encoder(w)
    return mdl.features(z, w), z


def linear_eval_graph(mdl, w: ComponentInputPayload) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute linear evaluation, dz, and states, z, for the model."""
    runtime = _require_graph_sequence_runtime(w)
    if not (runtime.is_uniform_length and runtime.is_fixed_topology):
        z = torch.stack(
            [mdl.encoder(runtime.get_step(step)) for step in range(runtime.n_steps)], dim=1
        )
        z_dot = torch.stack(
            [mdl.dynamics(z[:, step], runtime.get_step(step)) for step in range(runtime.n_steps)],
            dim=1,
        )
        return z_dot.permute(0, 2, 1, 3), z.permute(0, 2, 1, 3)
    z = mdl.encoder(w)
    z_dot = mdl.dynamics(z, w)
    return z_dot.permute(0, 2, 1, 3), z.permute(0, 2, 1, 3)


def linear_features_graph(mdl, w: ComponentInputPayload) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute linear features, f, and outputs, dz, for the model."""
    runtime = _require_graph_sequence_runtime(w)
    if not (runtime.is_uniform_length and runtime.is_fixed_topology):
        z = torch.stack(
            [mdl.encoder(runtime.get_step(step)) for step in range(runtime.n_steps)], dim=1
        )
        f = torch.stack(
            [mdl.features(z[:, step], runtime.get_step(step)) for step in range(runtime.n_steps)],
            dim=1,
        )
        return f.permute(0, 2, 1, 3), z.permute(0, 2, 1, 3)
    z = mdl.encoder(w)
    f = mdl.features(z, w)
    return f.permute(0, 2, 1, 3), z.permute(0, 2, 1, 3)


#: Mapping of linear evaluation and features functions.
LIN_MAP = {
    "smpl": (linear_eval_smpl, linear_features_smpl),
    "graph": (linear_eval_graph, linear_features_graph),
}
