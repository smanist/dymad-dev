import torch
import torch.nn as nn
from typing import Callable, Optional, Union

from dymad.modules.helpers import _resolve_activation, _resolve_init, INIT_MAP_W, INIT_MAP_B
from dymad.modules.linear import FlexLinear
from dymad.modules.misc import TakeFirst

class MLP(nn.Module):
    r"""
    Fully-connected feed-forward network

    Assuming the following architecture:

        in_dim -> (Linear -> Act) x n_hidden -> Linear -> out_dim

    Args:
        input_dim (int):
            Dimension of the input features.
        hidden_dim (int):
            Width of every hidden layer.
        output_dim (int):
            Dimension of the network output.
        n_layers (int, default = 2):
            Number of total layers.

            - If 0, same as Identity, or TakeFirst.
            - If 1, same as Linear.
            - If 2, same as `Linear -> activation -> Linear`.
            - Otherwise, hidden layers are inserted.

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
        hidden_dim: int,
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
            layers = [nn.Linear(input_dim, hidden_dim, dtype=dtype, device=device), _act()]
            for _ in range(n_layers - 2):
                layers += [nn.Linear(hidden_dim, hidden_dim, dtype=dtype, device=device), _act()]
            layers.append(nn.Linear(hidden_dim, output_dim, dtype=dtype, device=device))
            if end_activation:
                layers.append(_act())
            self.net = nn.Sequential(*layers)

        # Cache init kwargs for later use in self.apply
        self._weight_init = _resolve_init(weight_init, INIT_MAP_W)
        self._bias_init = _resolve_init(bias_init, INIT_MAP_B)

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
        if isinstance(m, FlexLinear):
            m._init_linear(self._weight_init, self._bias_init, self._gain)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ResBlockMLP(MLP):
    """
    Residual block with MLP as the nonlinearity.

    See `MLP` for the arguments.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 n_layers: int = 2,
                 activation: Union[str, nn.Module, Callable[[], nn.Module]] = nn.ReLU,
                 weight_init: Union[str, Callable[[torch.Tensor, float], None]] = nn.init.xavier_uniform_,
                 bias_init: Callable[[torch.Tensor], None] = nn.init.zeros_,
                 gain: Optional[float] = 1.0,
                 end_activation: bool = True,
                 dtype=None, device=None
                 ):
        assert input_dim == output_dim, "Input and output dimensions must match for ResBlock"
        super().__init__(input_dim, hidden_dim, output_dim,
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
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 n_layers: int = 2,
                 activation: Union[str, nn.Module, Callable[[], nn.Module]] = nn.ReLU,
                 weight_init: Union[str, Callable[[torch.Tensor, float], None]] = nn.init.xavier_uniform_,
                 bias_init: Callable[[torch.Tensor], None] = nn.init.zeros_,
                 gain: Optional[float] = 1.0,
                 end_activation: bool = True,
                 dtype=None, device=None):
        assert output_dim > input_dim, "Output dimension must be greater than input dimension"
        super().__init__(input_dim, hidden_dim, output_dim-input_dim,
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
