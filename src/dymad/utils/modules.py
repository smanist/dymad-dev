from functools import partial
from scipy.interpolate import interp1d
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch_geometric.nn.conv import MessagePassing
    from torch_geometric.nn import ChebConv, SAGEConv
except:
    MessagePassing = None
    ChebConv, SAGEConv = None, None
from typing import Callable, Optional, Union, Tuple

class TakeFirst(nn.Module):
    """
    Pass-through layer that returns the first `m` entries in the last axis.

    Args:
        m (int): Number of entries to take from the last axis.
    """
    def __init__(self, m: int):
        super().__init__()
        assert m > 0, "m must be a positive integer"
        self.m = m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """"""
        return x[..., :self.m] if x.ndim > 1 else x[:self.m]

    def diagnostic_info(self) -> str:
        return f"m: {self.m}"

class TakeFirstGraph(TakeFirst):
    """
    Graph version of TakeFirst.

    Input (..., n_nodes, n_features)
    Output (..., n_nodes*m)
    """
    def forward(self, x: torch.Tensor, edge_index, **kwargs) -> torch.Tensor:
        """"""
        out_shape = x.shape[:-2] + (-1,)
        return x[..., :self.m].reshape(*out_shape) if x.ndim > 1 else x[:self.m]

class FlexLinear(nn.Module):
    """
    A linear layer that can store weights either as a full matrix (MxN)
    or as low-rank factors (U, V) with efficient matvec operations.

    In the low-rank mode, the weight matrix is represented as:
        W = U @ V^T
    where U is (M x r) and V is (N x r).
    """
    def __init__(self, in_features, out_features, bias=True, dtype=None, device=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Full weight params
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype, device=device))
        self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device)) if bias else None

        # Low-rank params
        self.U = nn.Parameter(torch.empty(0, 0, dtype=dtype, device=device), requires_grad=False)
        self.V = nn.Parameter(torch.empty(0, 0, dtype=dtype, device=device), requires_grad=False)

        self.mode = "full"
        self.rank = None

    def _init_linear(
            self, 
            weight_init: Union[str, Callable[[torch.Tensor, float], None]] = nn.init.xavier_uniform_,
            bias_init: Callable[[torch.Tensor], None] = nn.init.zeros_,
            gain: float = 1.0) -> None:
        if self.mode == "full":
            weight_init(self.weight, gain)
        else:
            weight_init(self.U, gain)
            weight_init(self.V, gain)
        if self.bias is not None:
            bias_init(self.bias)

    @torch.no_grad()
    def set_full(self, W: torch.Tensor, b: torch.Tensor | None):
        """Switch to full mode and copy parameters."""
        assert W.shape == (self.out_features, self.in_features)
        self.mode = "full"
        self.rank = None
        # release low-rank
        self.U.requires_grad_(False)
        self.V.requires_grad_(False)
        self.U = nn.Parameter(torch.empty(0, 0, dtype=W.dtype, device=W.device), requires_grad=False)
        self.V = nn.Parameter(torch.empty(0, 0, dtype=W.dtype, device=W.device), requires_grad=False)
        # set full
        self.weight.data.copy_(W)
        if self.bias is not None and b is not None:
            self.bias.data.copy_(b)

    @torch.no_grad()
    def set_lora(self, U: torch.Tensor, V: torch.Tensor, b: torch.Tensor | None):
        """Switch to lowrank mode and copy factors. U: out*r, V: in*r."""
        assert U.shape[0] == self.out_features and V.shape[0] == self.in_features
        assert U.shape[1] == V.shape[1]
        self.rank = U.shape[1]
        self.mode = "lora"

        # freeze full weight (not used in forward)
        self.weight.requires_grad_(False)

        # (re)allocate factors with grad
        self.U = nn.Parameter(U.clone(), requires_grad=True)
        self.V = nn.Parameter(V.clone(), requires_grad=True)

        if self.bias is not None and b is not None:
            self.bias.data.copy_(b)

    @torch.no_grad()
    def set_weights(
        self,
        W: torch.Tensor | None = None, b: torch.Tensor | None = None,
        U: torch.Tensor | None = None, V: torch.Tensor | None = None):
        if W is not None:
            self.set_full(W, b)
        elif U is not None and V is not None:
            self.set_lora(U, V, b)
        else:
            raise ValueError("Must provide either full weights (W) or low-rank factors (U, V).")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "full":
            return F.linear(x, self.weight, self.bias)
        else:
            # Efficient matvec-only: (x @ V) @ U^T + b
            y = (x @ self.V) @ self.U.T
            if self.bias is not None:
                y = y + self.bias
            return y

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ):
        # U and V, if exist, are empty if full mode, otherwise they are low-rank factors.
        U_ckpt = state_dict.get(prefix + "U", None)
        V_ckpt = state_dict.get(prefix + "V", None)
        if U_ckpt is None or V_ckpt is None:
            is_lowrank = False
        else:
            is_lowrank = U_ckpt.shape[0] > 0 and V_ckpt.shape[0] > 0

        if is_lowrank:
            if self.U.shape != U_ckpt.shape or self.V.shape != V_ckpt.shape:
                # re-register parameters with correct shapes
                device = self.weight.device
                self.U = nn.Parameter(torch.empty_like(U_ckpt, dtype=U_ckpt.dtype, device=device), requires_grad=True)
                self.V = nn.Parameter(torch.empty_like(V_ckpt, dtype=V_ckpt.dtype, device=device), requires_grad=True)

        self.mode = "lora" if is_lowrank else "full"
        self.rank = self.U.shape[1] if is_lowrank else None

        # Now let the default loader copy tensors
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs
        )

_ACT_MAP = {
    # common aliases -> canonical class
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

_GCL_MAP = {
    # common aliases -> canonical class
    "sage"     : SAGEConv,
    "cheb"     : ChebConv
}

_INIT_MAP_W = {
    # aliases -> torch.nn.init functions
    "kaiming_uniform": nn.init.kaiming_uniform_,
    "kaiming_normal":  nn.init.kaiming_normal_,
    "xavier_uniform":  nn.init.xavier_uniform_,
    "xavier_normal":   nn.init.xavier_normal_,
    "orthogonal":      nn.init.orthogonal_,
    "normal":          nn.init.normal_,
    "trunc_normal":    nn.init.trunc_normal_,  # PyTorch ≥1.12
    "uniform":         nn.init.uniform_,
}

_INIT_MAP_B = {
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
        if key not in _ACT_MAP:
            raise ValueError(f"Unknown activation string '{spec}'. "
                             f"Valid keys are {sorted(_ACT_MAP.keys())}.")
        if key == "prelu":
            # dtype of the slope
            return partial(_ACT_MAP[key], dtype=dtype, device=device)
        return _ACT_MAP[key]

    # case 2 ─ activation *class* (subclass of nn.Module)
    if isinstance(spec, type) and issubclass(spec, nn.Module):
        return spec

    # case 3 ─ already-constructed module
    if isinstance(spec, nn.Module):
        return type(spec)

    raise TypeError("activation must be str, nn.Module subclass, "
                    f"or nn.Module instance, got {type(spec)}")

def _resolve_gcl(spec) -> nn.Module:
    """
    Turn a user-supplied graph convolutional layer *specification* into an nn.Module.
    `spec` can be a string, a GCL class, or a constructed module.
    """
    # case 1 ─ string
    if isinstance(spec, str):
        key = spec.lower()
        if key not in _GCL_MAP:
            raise ValueError(f"Unknown GCL string '{spec}'. "
                             f"Valid keys are {sorted(_GCL_MAP.keys())}.")
        return _GCL_MAP[key]

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

class MLP(nn.Module):
    r"""
    Fully-connected feed-forward network

    Assuming the following architecture:

        in_dim -> (Linear -> Act) x n_latent -> Linear -> out_dim

    Args:
        input_dim (int):
            Dimension of the input features.
        latent_dim (int):
            Width of every hidden layer.
        output_dim (int):
            Dimension of the network output.
        n_layers (int, default = 2):
            Number of total layers.

            - If 0, same as Identity, or TakeFirst.
            - If 1, same as Linear.
            - If 2, same as `Linear -> activation -> Linear`.
            - Otherwise, latent layers are inserted.

        activation (nn.Module or Callable[[], nn.Module], default = nn.ReLU):
            Non-linearity to insert after every hidden Linear.
            Pass either a class (e.g. `nn.Tanh`) or an already-constructed module.
        weight_init (Callable[[torch.Tensor, float], None], default = `nn.init.kaiming_uniform_`):
            Function used to initialise each Linear layer's *weight* tensor.
            Must accept `(tensor, gain)` signature like the functions in
            `torch.nn.init`.
        bias_init (Callable[[torch.Tensor], None], default = `nn.init.zeros_`):
            Function used to initialise each Linear layer's *bias* tensor.
        gain (Optional[float], default = 1.0):
            In the linear layers, the weights are initialised with the standard
            `nn.init.calculate_gain(<nonlinearity>)`
            Gain is multiplied to the calculated gain.  By default gain=1, so no change.
        end_activation (bool, default = True):

            - If ``True``, the last layer is followed by an activation function.
            - Otherwise, the last layer is a plain Linear layer.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        output_dim: int,
        *,
        n_layers: int = 2,
        activation: Union[str, nn.Module, Callable[[], nn.Module]] = nn.ReLU,
        weight_init: Union[str, Callable[[torch.Tensor, float], None]] = nn.init.xavier_uniform_,
        bias_init: Callable[[torch.Tensor], None] = nn.init.zeros_,
        gain: Optional[float] = 1.0,
        end_activation: bool = True,
        dtype=None, device=None
    ):
        super().__init__()

        _act = _resolve_activation(activation, dtype, device)

        if n_layers == 0:
            if input_dim == output_dim:
                self.net = nn.Identity()
            else:
                self.net = TakeFirst(output_dim)
        elif n_layers == 1:
            if end_activation:
                self.net = nn.Sequential(
                    nn.Linear(input_dim, output_dim, dtype=dtype, device=device),
                    _act()
                )
            else:
                self.net = nn.Linear(input_dim, output_dim, dtype=dtype, device=device)
        else:
            layers = [nn.Linear(input_dim, latent_dim, dtype=dtype, device=device), _act()]
            for _ in range(n_layers - 2):
                layers += [nn.Linear(latent_dim, latent_dim, dtype=dtype, device=device), _act()]
            layers.append(nn.Linear(latent_dim, output_dim, dtype=dtype, device=device))
            if end_activation:
                layers.append(_act())
            self.net = nn.Sequential(*layers)

        # Cache init kwargs for later use in self.apply
        self._weight_init = _resolve_init(weight_init, _INIT_MAP_W)
        self._bias_init = _resolve_init(bias_init, _INIT_MAP_B)

        # Compute gain
        act_name = _act().__class__.__name__.lower()
        _g = nn.init.calculate_gain(act_name if act_name not in ["gelu", "prelu", "identity"] else "relu")
        self._gain = gain*_g

        # Initialise weights & biases
        self.apply(self._init_linear)

    def diagnostic_info(self) -> str:
        return f"Weight init: {self._weight_init}, " + \
               f"Weight gain: {self._gain}, " + \
               f"Bias init: {self._bias_init}"

    def _init_linear(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            self._weight_init(m.weight, self._gain)
            self._bias_init(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ResBlockMLP(MLP):
    """
    Residual block with MLP as the nonlinearity.

    See `MLP` for the arguments.
    """
    def __init__(self, input_dim: int, latent_dim: int, output_dim: int,
                 n_layers: int = 2,
                 activation: Union[str, nn.Module, Callable[[], nn.Module]] = nn.ReLU,
                 weight_init: Union[str, Callable[[torch.Tensor, float], None]] = nn.init.xavier_uniform_,
                 bias_init: Callable[[torch.Tensor], None] = nn.init.zeros_,
                 gain: Optional[float] = 1.0,
                 end_activation: bool = True,
                 dtype=None, device=None
                 ):
        assert input_dim == output_dim, "Input and output dimensions must match for ResBlock"
        super().__init__(input_dim, latent_dim, output_dim,
                         n_layers=n_layers,
                         activation=activation,
                         weight_init=weight_init,
                         bias_init=bias_init,
                         gain=gain,
                         end_activation=end_activation,
                         dtype=dtype,
                         device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the residual block.

        Args:
            x (torch.Tensor): Input tensor of shape (..., input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (..., output_dim).
        """
        return x + self.net(x)

class IdenCatMLP(MLP):
    """
    Identity concatenation MLP.

    This MLP concatenates the input with the output of the MLP.

    Note:
        The output dimension represents the **total** output features and must be greater than the input dimension.

    See `MLP` for the arguments.
    """
    def __init__(self, input_dim: int, latent_dim: int, output_dim: int,
                 n_layers: int = 2,
                 activation: Union[str, nn.Module, Callable[[], nn.Module]] = nn.ReLU,
                 weight_init: Union[str, Callable[[torch.Tensor, float], None]] = nn.init.xavier_uniform_,
                 bias_init: Callable[[torch.Tensor], None] = nn.init.zeros_,
                 gain: Optional[float] = 1.0,
                 end_activation: bool = True,
                 dtype=None, device=None):
        assert output_dim > input_dim, "Output dimension must be greater than input dimension"
        super().__init__(input_dim, latent_dim, output_dim-input_dim,
                         n_layers=n_layers,
                         activation=activation,
                         weight_init=weight_init,
                         bias_init=bias_init,
                         gain=gain,
                         end_activation=end_activation,
                         dtype=dtype,
                         device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the identity concatenation MLP.

        Args:
            x (torch.Tensor): Input tensor of shape (..., input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (..., output_dim).
        """
        return torch.cat([x, self.net(x)], dim=-1)

class GNN(nn.Module):
    """
    Configurable Graph Neural Network using a choice of GCL (e.g., SAGEConv, ChebConv) and activations.

    Due to the implementation, the GNN is applied sequentially to batch data.

    To interface with other parts of the code, the model assumes the input to be node-wise, (..., n_nodes, n_input),
    but the output is reshaped to concatenate features across nodes, (..., n_nodes * n_output).
    See `forward` method for details.

    Args:
        input_dim (int): Dimension of input node features.
        latent_dim (int): Dimension of hidden layers.
        output_dim (int): Dimension of output node features.
        n_layers (int): Number of GCL layers.
        gcl (str | nn.Module | type, default='sage'): Graph convolution layer type or instance.
        activation (str | nn.Module | type, default='prelu'): Activation function.
        weight_init (str | callable, default='xavier_uniform'): Weight initializer.
        bias_init (str | callable, default='zeros'): Bias initializer.
        gain (float, default=1.0): Extra gain modifier for weight initialization.
        end_activation (bool, default=True): Whether to apply activation after last layer.
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        output_dim: int,
        n_layers: int,
        *,
        gcl: Union[str, nn.Module, type] = 'sage',
        activation: Union[str, nn.Module, Callable[[], nn.Module]] = 'prelu',
        weight_init: Union[str, Callable[[torch.Tensor, float], None]] = 'xavier_uniform',
        bias_init: Union[str, Callable[[torch.Tensor], None]] = 'zeros',
        gain: float = 1.0,
        end_activation: bool = True,
        dtype=None, device=None
    ):
        super().__init__()
        assert n_layers > 0, "n_layers must be a positive integer"

        _gcl = _resolve_gcl(gcl)
        _act = _resolve_activation(activation, dtype, device)
        self._weight_init = _resolve_init(weight_init, _INIT_MAP_W)
        self._bias_init = _resolve_init(bias_init, _INIT_MAP_B)

        act_name = _act().__class__.__name__.lower()
        _g = nn.init.calculate_gain(act_name if act_name not in ["gelu", "prelu", "identity"] else "relu")
        self._gain = gain * _g

        layers = []
        for i in range(n_layers):
            in_dim = input_dim if i == 0 else latent_dim
            out_dim = output_dim if i == n_layers - 1 else latent_dim
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
            self._weight_init(m.weight, self._gain)
        if hasattr(m, 'bias') and m.bias is not None:
            self._bias_init(m.bias)

    def forward(self, x, edge_index, **kwargs):
        """
        Forward pass through the GNN.

        `x` is of shape (n_batch, ..., n_nodes, n_features).
        `edge_index` is of shape (n_batch, 2, n_edges).

        The code returns a tensor of shape (n_batch, ..., n_nodes * n_features).

        If n_batch=1, we can process the entire batch in one go.
        Otherwise, we process each edge_index sequentially, which would incur
        a severe performance degradation.

        Fortunately, the inputs are assumed to come from DynGeoData, which
        collates batch data into a single large graph, so that here we always
        have n_batch=1.
        """
        if edge_index.shape[0] == 1:
            # The usual case, where we have a single edge_index
            return self._forward_single(x, edge_index[0], **kwargs)
        else:
            # The slow case, where we process multiple edge_indices sequentially
            assert len(x) == len(edge_index), \
                "Batch size of x and edge_index must match. Got {} and {}.".format(x.shape, edge_index.shape)
            tmp = []
            for _x, _e in zip(x, edge_index):
                tmp.append(self._forward_single(_x, _e, **kwargs))
            return torch.stack(tmp, dim=0)

    def _forward_single(self, x, edge_index, **kwargs):
        """
        Forward pass for one edge_index.
        """
        out_shape = x.shape[:-2] + (-1,)
        for layer in self.layers:
            if isinstance(layer, MessagePassing):
                x = layer(x, edge_index, **kwargs)
            else:
                x = layer(x)
        return x.reshape(*out_shape)

class ResBlockGNN(GNN):
    """
    Residual block with GNN as the nonlinearity.

    See `GNN` for the arguments.
    """
    def __init__(self, input_dim: int, latent_dim: int, output_dim: int,
                 n_layers: int,
                 gcl: Union[str, nn.Module, type] = 'sage',
                 activation: Union[str, nn.Module, Callable[[], nn.Module]] = 'prelu',
                 weight_init: Union[str, Callable[[torch.Tensor, float], None]] = 'xavier_uniform',
                 bias_init: Callable[[torch.Tensor], None] = 'zeros',
                 gain: float = 1.0,
                 end_activation: bool = True,
                 dtype=None, device=None):
        assert input_dim == output_dim, "Input and output dimensions must match for ResBlock"
        super().__init__(input_dim, latent_dim, output_dim,
                         n_layers=n_layers,
                         gcl=gcl,
                         activation=activation,
                         weight_init=weight_init,
                         bias_init=bias_init,
                         gain=gain,
                         end_activation=end_activation,
                         dtype=dtype,
                         device=device)

    def forward(self, x, edge_index, **kwargs):
        inp_shape = x.shape[:-1] + (-1,)
        out_shape = x.shape[:-2] + (-1,)
        res = x + super().forward(x, edge_index, **kwargs).reshape(*inp_shape)
        return res.reshape(*out_shape)

class IdenCatGNN(GNN):
    """
    Identity concatenation GNN.

    This GNN concatenates the input with the output of the GNN.

    Note:
        The output dimension represents the **total** output features and must be greater than the input dimension.

    See `GNN` for the arguments.
    """
    def __init__(self, input_dim: int, latent_dim: int, output_dim: int,
                 n_layers: int,
                 gcl: Union[str, nn.Module, type] = 'sage',
                 activation: Union[str, nn.Module, Callable[[], nn.Module]] = 'prelu',
                 weight_init: Union[str, Callable[[torch.Tensor, float], None]] = 'xavier_uniform',
                 bias_init: Callable[[torch.Tensor], None] = 'zeros',
                 gain: float = 1.0,
                 end_activation: bool = True,
                 dtype=None, device=None):
        assert output_dim > input_dim, "Output dimension must be greater than input dimension"
        super().__init__(input_dim, latent_dim, output_dim-input_dim,
                         n_layers=n_layers,
                         gcl=gcl,
                         activation=activation,
                         weight_init=weight_init,
                         bias_init=bias_init,
                         gain=gain,
                         end_activation=end_activation,
                         dtype=dtype,
                         device=device)

    def forward(self, x, edge_index, **kwargs):
        inp_shape = x.shape[:-1] + (-1,)
        out_shape = x.shape[:-2] + (-1,)
        tmp = super().forward(x, edge_index, **kwargs).reshape(*inp_shape)
        out = torch.cat([x, tmp], dim=-1)
        return out.reshape(*out_shape)

def make_autoencoder(
        type: str,
        input_dim: int, latent_dim: int, hidden_dim: int, enc_depth: int, dec_depth: int,
        output_dim: int = None, **kwargs) -> Tuple[nn.Module, nn.Module]:
    """
    Factory function to create preset autoencoder models. Including:

    - [mlp_smp] Simple version: MLP-in MLP-out
    - [mlp_res] Simple version but with ResBlockMLP
    - [mlp_cat] Concatenation as encoder [x MLP(x)], then TakeFirst as decoder
    - The graph version of the above: gnn_smp, gnn_res, gnn_cat

    Args:
        type (str): Type of autoencoder to create.
            One of {'mlp_smp', 'mlp_res', 'mlp_cat', 'gnn_smp', 'gnn_res', 'gnn_cat'}.
        input_dim (int): Dimension of the input features.
        latent_dim (int): Width of the latent layers (not the encoded space).
        hidden_dim (int): Dimension of the encoded space.
        enc_depth (int): Number of layers in the encoder.
        dec_depth (int): Number of layers in the decoder.
        output_dim (int, optional): Dimension of the output features, defaults to `input_dim`.
        **kwargs: Additional keyword arguments passed to the MLP or GNN constructors.
    """
    # Prepare the arguments
    if output_dim is None:
        output_dim = input_dim

    encoder_args = dict(
        input_dim=input_dim,
        latent_dim=latent_dim,
        output_dim=hidden_dim,
        n_layers=enc_depth,
    )
    encoder_args.update(kwargs)
    decoder_args = dict(
        input_dim=hidden_dim,
        latent_dim=latent_dim,
        output_dim=output_dim,
        n_layers=dec_depth,
    )
    decoder_args.update(kwargs)

    # Generate the encoder and decoder based on the type
    _type = type.lower()
    encoder, decoder = None, None
    if _type[:3] == "mlp":
        if _type == "mlp_smp":
            encoder = MLP(**encoder_args)
            decoder = MLP(**decoder_args)

        elif _type == "mlp_res":
            encoder = ResBlockMLP(**encoder_args)
            decoder = ResBlockMLP(**decoder_args)

        elif _type == "mlp_cat":
            encoder = IdenCatMLP(**encoder_args)
            decoder = TakeFirst(output_dim)

    elif _type[:3] == "gnn":
        if _type == "gnn_smp":
            encoder = GNN(**encoder_args)
            decoder = GNN(**decoder_args)

        elif _type == "gnn_res":
            encoder = ResBlockGNN(**encoder_args)
            decoder = ResBlockGNN(**decoder_args)

        elif _type == "gnn_cat":
            encoder = IdenCatGNN(**encoder_args)
            decoder = TakeFirstGraph(output_dim)

    if encoder is None or decoder is None:
        raise ValueError(f"Unknown autoencoder type '{type}'.")

    return encoder, decoder

class ControlInterpolator(nn.Module):
    """
    Interpolates the sampled control signal u(t_k) when the ODE solver
    requests u(t_query).

    Args:
        t (torch.Tensor): 1-D tensor of shape (N,). Sampling times (must be ascending).
        u (torch.Tensor): Tensor of shape (..., N, m). Control samples, m inputs per step.
        order (str): Interpolation mode. One of {'zoh', 'linear', 'cubic', etc}.

    Note:
        Not to be confused with `dymad.utils.sampling._build_interpolant`,
        which is for data generation, esp. with Numpy.
        `ControlInterpolator` is meant to be used in a Torch setting.
    """
    def __init__(self, t, u, order='linear'):
        super().__init__()

        assert u.ndim >= 2, "Control signal must have at least 2 dimensions"

        self.order = order.lower()
        self.register_buffer('t', t)
        self.register_buffer('u', u)

        if self.order == 'zoh':
            self._interp = self._interp_0
        elif self.order == 'linear':
            self._interp = self._interp_1
        else:
            # Assuming option for 'scipy' interpolation
            self._cpu_t  = t.detach().cpu().numpy()
            self._cpu_u  = u.detach().cpu().numpy()
            self._spl    = interp1d(self._cpu_t,
                                    self._cpu_u,
                                    kind=order,
                                    axis=-2,
                                    fill_value="extrapolate",
                                    assume_sorted=True)
            self._interp = self._interp_s

    def forward(self, t_query: torch.Tensor) -> torch.Tensor:
        return self._interp(t_query)

    def _interp_0(self, t_query: torch.Tensor) -> torch.Tensor:
        idx = torch.searchsorted(self.t, t_query).clamp(1, self.t.numel()-1)
        return self.u[..., idx-1, :]

    def _interp_1(self, t_query: torch.Tensor) -> torch.Tensor:
        idx = torch.searchsorted(self.t, t_query).clamp(1, self.t.numel()-1)
        t0, t1   = self.t[idx-1], self.t[idx]
        u0, u1   = self.u[..., idx-1, :], self.u[..., idx, :]
        w        = (t_query - t0) / (t1 - t0)
        return (1. - w) * u0 + w * u1

    def _interp_s(self, t_query: torch.Tensor) -> torch.Tensor:
        uq = self._spl(t_query.detach().cpu().numpy())
        return torch.as_tensor(uq,
                                device=t_query.device,
                                dtype=self.u.dtype)
