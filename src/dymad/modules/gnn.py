import torch
import torch.nn as nn
try:
    from torch_geometric.nn.conv import ChebConv, GATConv, GCNConv, GraphConv, MessagePassing, SAGEConv
except:
    MessagePassing = None
from typing import Callable, Union

from dymad.modules.helpers import _resolve_activation, _resolve_gcl, _resolve_init, INIT_MAP_W, INIT_MAP_B

class GNN(nn.Module):
    """
    Configurable Graph Neural Network using a choice of GCL (e.g., SAGEConv, ChebConv) and activations.

    Due to the implementation, the GNN is applied sequentially to batch data.

    To interface with other parts of the code, the model assumes the input to be node-wise, (..., n_nodes, n_input),
    but the output is reshaped to concatenate features across nodes, (..., n_nodes * n_output).
    See `forward` method for details.

    Args:
        input_dim (int): Dimension of input node features.
        hidden_dim (int): Dimension of hidden layers.
        output_dim (int): Dimension of output node features.
        n_layers (int): Number of GCL layers.
        gcl (str | nn.Module | type, default='sage'): Graph convolution layer type or instance.
        gcl_opts (dict, default={}): Options passed to the GCL constructor.
        activation (str | nn.Module | type, default='prelu'): Activation function.
        weight_init (str | callable, default='xavier_uniform'): Weight initializer.
        bias_init (str | callable, default='zeros'): Bias initializer.
        gain (float, default=1.0): Extra gain modifier for weight initialization.
        end_activation (bool, default=True): Whether to apply activation after last layer.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int,
        *,
        gcl: Union[str, nn.Module, type] = 'sage',
        gcl_opts: dict = {},
        activation: Union[str, nn.Module, Callable[[], nn.Module]] = 'prelu',
        weight_init: Union[str, Callable[[torch.Tensor, float], None]] = 'xavier_uniform',
        bias_init: Union[str, Callable[[torch.Tensor], None]] = 'zeros',
        gain: float = 1.0,
        end_activation: bool = True,
        dtype=None, device=None
    ):
        super().__init__()

        _gcl = _resolve_gcl(gcl, gcl_opts)
        _act = _resolve_activation(activation, dtype, device)
        self._weight_init = _resolve_init(weight_init, INIT_MAP_W)
        self._bias_init = _resolve_init(bias_init, INIT_MAP_B)

        act_name = _act().__class__.__name__.lower()
        _g = nn.init.calculate_gain(act_name if act_name not in ["gelu", "prelu", "identity"] else "relu")
        self._gain = gain * _g

        if n_layers == 0:
            self.layers = nn.ModuleList()  # Identity mapping
        else:
            layers = []
            for i in range(n_layers):
                in_dim = input_dim if i == 0 else hidden_dim
                out_dim = output_dim if i == n_layers - 1 else hidden_dim
                # Each GCL layer can be a new instance
                gcl_layer = _gcl(in_dim, out_dim)
                layers.append(gcl_layer)
                # Only add activation if not last layer or end_activation is True
                if i < n_layers - 1 or end_activation:
                    # Each activation can be a new instance
                    layers.append(_act())
            self.layers = nn.ModuleList(layers)

        self.apply(self._init_gcl)

    def diagnostic_info(self) -> str:
        return f"Weight init: {self._weight_init}, " + \
               f"Weight gain: {self._gain}, " + \
               f"Bias init: {self._bias_init}"

    def _init_gcl(self, m: nn.Module) -> None:
        # Only initialize GCL layers with weight/bias
        if hasattr(m, 'weight') and m.weight is not None:
            if m.weight.ndim >= 2:
                self._weight_init(m.weight, self._gain)
        if hasattr(m, 'bias') and m.bias is not None:
            self._bias_init(m.bias)

    def forward(self, x, edge_index, edge_weights, edge_attr, **kwargs):
        """
        Forward pass through the GNN.

        - `x` (..., n_nodes, n_features).
        - `edge_index` (..., n_edges, 2).
        - `edge_weights` (..., n_edges).
        - `edge_attr` (..., n_edges, n_edge_features).
        - Returns (..., n_nodes*n_new_features).

        If ...=1, we can process the entire batch in one go.
        Otherwise, we aggregate the graph on the fly so the shapes are reduced
        to the first case.  The aggregation takes a bit more time.
        """
        assert edge_index.ndim == 3, "edge_index must have shape (..., n_edges, 2)"
        if x.shape[0] == 1:
            # The usual case, where we have a single edge_index
            ew = None if edge_weights is None else edge_weights[0]
            ea = None if edge_attr is None else edge_attr[0]
            return self._forward_single(x, edge_index[0].transpose(0, 1), ew, ea, **kwargs)
        else:
            # The slower case, where we aggregate graph on the fly
            _x_batch, _x_shape = x.shape[:-2], x.shape[-2:]
            _e_batch, _e_shape = edge_index.shape[:-2], edge_index.shape[-2:]
            assert _x_batch == _e_batch, \
                f"Batch shape of x and edge_index must match. Got {_x_batch} and {_e_batch}."

            # Aggregate graph by shifting node indices
            _ei = edge_index.unbind()
            _tmp = 1
            for d in _x_batch:
                _tmp *= d
            _n_nodes = [0] + [_x_shape[0]] * (_tmp - 1)
            _offset = torch.tensor(_n_nodes).cumsum(dim=0)
            _ei_cat = torch.concatenate([
                b + _offset[i] for i, b in enumerate(_ei)],
                dim=-2).transpose(0, 1)

            if edge_weights is None:
                ew = None
            else:
                ew = edge_weights.values()

            if edge_attr is None:
                ea = None
            else:
                ea = edge_attr.values()

            # Process node features
            _x_cat = x.reshape(1, -1, _x_shape[1])
            _out = self._forward_single(_x_cat, _ei_cat, ew, ea, **kwargs)
            return _out.reshape(*_x_batch, -1)

    def _forward_single(self, x, edge_index, edge_weights, edge_attr, **kwargs):
        """
        Forward pass for one edge_index.
        """
        out_shape = x.shape[:-2] + (-1,)
        for layer in self.layers:
            if isinstance(layer, SAGEConv):
                x = layer(x, edge_index, **kwargs)
            elif isinstance(layer, (ChebConv, GCNConv, GraphConv)):
                x = layer(x, edge_index, edge_weight=edge_weights, **kwargs)
            elif isinstance(layer, GATConv):
                x = layer(x, edge_index, edge_attr=edge_attr, **kwargs)
            elif isinstance(layer, MessagePassing):
                x = layer(x, edge_index, edge_weight=edge_weights, edge_attr=edge_attr, **kwargs)
            else:
                x = layer(x)
        return x.reshape(*out_shape)

class ResBlockGNN(GNN):
    """
    Residual block with GNN as the nonlinearity.

    See `GNN` for the arguments.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 n_layers: int,
                 gcl: Union[str, nn.Module, type] = 'sage',
                 gcl_opts: dict = {},
                 activation: Union[str, nn.Module, Callable[[], nn.Module]] = 'prelu',
                 weight_init: Union[str, Callable[[torch.Tensor, float], None]] = 'xavier_uniform',
                 bias_init: Callable[[torch.Tensor], None] = 'zeros',
                 gain: float = 1.0,
                 end_activation: bool = True,
                 dtype=None, device=None):
        assert input_dim == output_dim, "Input and output dimensions must match for ResBlock"
        super().__init__(input_dim, hidden_dim, output_dim,
                         n_layers=n_layers,
                         gcl=gcl,
                         gcl_opts=gcl_opts,
                         activation=activation,
                         weight_init=weight_init,
                         bias_init=bias_init,
                         gain=gain,
                         end_activation=end_activation,
                         dtype=dtype,
                         device=device)

    def forward(self, x, edge_index, edge_weights, edge_attr, **kwargs):
        inp_shape = x.shape[:-1] + (-1,)
        out_shape = x.shape[:-2] + (-1,)
        res = x + super().forward(x, edge_index, edge_weights, edge_attr, **kwargs).reshape(*inp_shape)
        return res.reshape(*out_shape)

class IdenCatGNN(GNN):
    """
    Identity concatenation GNN.

    This GNN concatenates the input with the output of the GNN.

    Note:
        The output dimension represents the **total** output features and must be greater than the input dimension.

    See `GNN` for the arguments.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 n_layers: int,
                 gcl: Union[str, nn.Module, type] = 'sage',
                 gcl_opts: dict = {},
                 activation: Union[str, nn.Module, Callable[[], nn.Module]] = 'prelu',
                 weight_init: Union[str, Callable[[torch.Tensor, float], None]] = 'xavier_uniform',
                 bias_init: Callable[[torch.Tensor], None] = 'zeros',
                 gain: float = 1.0,
                 end_activation: bool = True,
                 dtype=None, device=None):
        assert output_dim > input_dim, "Output dimension must be greater than input dimension"
        super().__init__(input_dim, hidden_dim, output_dim-input_dim,
                         n_layers=n_layers,
                         gcl=gcl,
                         gcl_opts=gcl_opts,
                         activation=activation,
                         weight_init=weight_init,
                         bias_init=bias_init,
                         gain=gain,
                         end_activation=end_activation,
                         dtype=dtype,
                         device=device)

    def forward(self, x, edge_index, edge_weights, edge_attr, **kwargs):
        inp_shape = x.shape[:-1] + (-1,)
        out_shape = x.shape[:-2] + (-1,)
        tmp = super().forward(x, edge_index, edge_weights, edge_attr, **kwargs).reshape(*inp_shape)
        out = torch.cat([x, tmp], dim=-1)
        return out.reshape(*out_shape)
