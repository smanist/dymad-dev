import torch
import torch.nn as nn
from typing import Callable, Union, Optional

from dymad.modules.helpers import _resolve_activation, _resolve_init, INIT_MAP_W, INIT_MAP_B


class SequentialBase(nn.Module):
    """Interface module that handles time-delayed input sequences.

    The module assumes that the input is given in shape (..., seq_len * input_dim),
    where seq_len is the length of the time-delay/sequence, and input_dim is the
    dimension of each step's input features.  The module reshapes the input to
    (..., seq_len, input_dim) and passes it to an internal sequential model (e.g., RNN),
    and return the output either at the last step (..., output_dim) or
    the full sequence flattened (..., seq_len * output_dim).
    
    It considers two types of architectures

    - Internally construct RNN-like models, that applies to the input in the standard way.
      This is usually good for defining dynamics.
    - Externally provided models, that process step by step and not necessarily recurrently.
      This is usually good for defining encoders/decoders.  Examples are MLP and GNN.
    - In both cases, the models expect input of shape (-1, seq_len, input_dim) and return
      output of shape (-1, seq_len, output_dim).
    - In either case, a subclass must implement `_run_seq()` method that defines how to run the model.

    The module can also operate in two modes

    - last_only=True: returns only the output at the last step (..., output_dim).
      Usually for dynamics.
    - last_only=False: returns the outputs at all steps, flattened (..., seq_len * output_dim).
      Usually for encoders/decoders.

    Args:
        seq_len (int): Length of the input sequences.
        last_only (Optional[bool]): Whether to return only the last step output, default is True.
        net (Optional[nn.Module]): Optional externally provided model.  If None,
          an internal RNN-like model is constructed.
        input_dim (Optional[int]): Dimension of the input features of all steps (for RNN-like).
        hidden_dim (Optional[int]): Width of the hidden layers (for RNN-like).
        output_dim (Optional[int]): Dimension of the output features of all steps (for RNN-like).
        n_layers (Optional[int]): Number of layers (for RNN-like).
        activation (Union[str, nn.Module, Callable[[], nn.Module]]): Activation function (for RNN-like).
        weight_init (Union[str, Callable[[torch.Tensor, float], None]]): Weight initialization method (for RNN-like).
        bias_init (Callable[[torch.Tensor], None]): Bias initialization method (for RNN-like).
        gain (Optional[float]): Gain factor for weight initialization (for RNN-like).
        dtype: Data type for the module (for RNN-like).
        device: Device for the module (for RNN-like).
        **kwargs: Additional keyword arguments passed to the internal model constructor.
    """

    def __init__(
        self,
        seq_len: int,
        *,
        last_only: Optional[bool] = True,
        net: Optional[nn.Module] = None,
        input_dim: Optional[int] = -1,
        hidden_dim: Optional[int] = -1,
        output_dim: Optional[int] = -1,
        n_layers: Optional[int] = 2,
        activation: Union[str, nn.Module, Callable[[], nn.Module]] = nn.ReLU,
        weight_init: Union[str, Callable[[torch.Tensor, float], None]] = nn.init.xavier_uniform_,
        bias_init: Callable[[torch.Tensor], None] = nn.init.zeros_,
        gain: Optional[float] = 1.0,
        dtype=None, device=None,
        **kwargs
    ):
        super().__init__()
        self.seq_len = seq_len
        self.last_only = last_only

        if net is not None:
            # Externally provided model
            self.net = net
            self.dtype = dtype
            self.device = device
            return

        # Check dimensions
        assert input_dim % seq_len == 0, f"input_dim {input_dim} must be divisible by seq_len {seq_len}."
        _inp_dim = input_dim // seq_len
        if last_only:
            _out_dim = output_dim
        else:
            assert output_dim % seq_len == 0, f"output_dim {output_dim} must be divisible by seq_len {seq_len} when last_only is False."
            _out_dim = output_dim // seq_len

        # Internally construct RNN-like model
        _act = _resolve_activation(activation, dtype, device)
        self._build_seq(_inp_dim, hidden_dim, _out_dim, n_layers, _act, dtype, device, **kwargs)

        # Cache init kwargs for later use in self.apply
        self._weight_init = _resolve_init(weight_init, INIT_MAP_W)
        self._bias_init = _resolve_init(bias_init, INIT_MAP_B)

        # Compute gain
        act_name = _act().__class__.__name__.lower()
        _g = nn.init.calculate_gain(act_name if act_name not in ["gelu", "prelu", "identity"] else "relu")
        self._gain = gain*_g

        # Initialise weights & biases
        self.apply(self._init_linear)

    def _init_linear(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            self._weight_init(m.weight, self._gain)
            self._bias_init(m.bias)

    def _build_seq(self, input_dim, hidden_dim, output_dim, n_layers, _act, dtype, device, **kwargs):
        """Build the internal sequential module.

        Expect input_dim and output_dim as the dimensions of the input/output features per step.
        """
        raise NotImplementedError("_build_seq must be implemented in subclasses.")

    def _run_seq(self, x_seq: torch.Tensor) -> torch.Tensor:
        """Evaluate the recurrent model.

        Expects input of shape (batch_size, seq_len, input_dim) and returns output of shape
        (batch_size, seq_len, output_dim).

        Depending on last_only, the `forward()` method will return either the last step output
        or the full sequence output flattened.
        """
        raise NotImplementedError("_run_seq must be implemented in subclasses.")

    def forward(self, x: torch.Tensor, u: torch.Tensor | None = None):
        """
        The network does concatenation internally, because x and u are
        both time-delayed and concatenated, and applying the sequential model
        requires to stack x and u of the same steps and then concatenate.

        Args:
            x: Stacked input tensor of shape (..., seq_len * x_dim)
            u: (Optional) Stacked control tensor of shape (..., seq_len * u_dim);
              needed when serving as encoder with inputs.

        Returns:
            output: Stacked output tensor of shape (..., output_dim),
                    where the last slot is the output of the sequential model at the last step
                    if last_only is True, otherwise all outputs are concatenated.
        """
        # Infer dimensions from stacked input
        batch_size = x.shape[:-1]
        input_dim = x.shape[-1] // self.seq_len

        # Reshape from stacked to sequence format
        x_seq = x.reshape(-1, self.seq_len, input_dim)
        if u is not None:
            u_dim = u.shape[-1] // self.seq_len
            u_seq = u.reshape(-1, self.seq_len, u_dim)
            x_seq = torch.cat([x_seq, u_seq], dim=-1)

        # Forward through the sequential model
        z = self._run_seq(x_seq)

        if self.last_only:
            z = z[..., -1, :]
        else:
            z = z.reshape(*batch_size, -1)

        return z


class VanillaRNN(SequentialBase):
    """Vanilla RNN from pytorch."""

    def _build_seq(self, input_dim, hidden_dim, output_dim, n_layers, _act, dtype, device, **kwargs):
        _act_name = _act().__class__.__name__.lower()
        assert _act_name in ['tanh', 'relu'], "Only 'tanh' and 'relu' activations are supported for nn.RNN."
        assert output_dim == hidden_dim, "For VanillaRNN, output_dim must equal hidden_dim."
        self.net = nn.RNN(
            input_size = input_dim,
            hidden_size = hidden_dim,
            num_layers = n_layers,
            nonlinearity = _act_name,
            batch_first = True,
            device=device,
            dtype=dtype,
            **kwargs
        )
        self.dtype = dtype
        self.device = device

    def _run_seq(self, x):
        output, _ = self.net(x, hidden=None)
        return output


class SimpleRNN(SequentialBase):
    """A simple recurrent neural network module
    
    One layer, unidirectional, but supports arbitrary activations with a linear readout.
    """

    def _build_seq(self, input_dim, hidden_dim, output_dim, n_layers, _act, dtype, device, **kwargs):
        assert n_layers == 1, f"SimpleRNN only supports n_layers=1, got {n_layers}"

        self.hidden_dim = hidden_dim

        # Linear transformations
        self.i2h = nn.Linear(input_dim,  hidden_dim, device=device, dtype=dtype)
        self.h2h = nn.Linear(hidden_dim, hidden_dim, device=device, dtype=dtype)
        self.h2o = nn.Linear(hidden_dim, output_dim, device=device, dtype=dtype)
        self.activation = _act()
        self.dtype = dtype
        self.device = device

    def _run_seq(self, x):
        hidden = torch.zeros(x.shape[0], self.hidden_dim, device=self.device, dtype=self.dtype)

        output = []
        for t in range(self.seq_len):
            hidden = self.activation(self.i2h(x[:, t, :]) + self.h2h(hidden))
            output.append(self.activation(self.h2o(hidden)))

        return torch.stack(output, dim=-2)


class StepwiseModel(SequentialBase):
    """Naive application of a network to each step of a sequence."""

    def _run_seq(self, x):
        return self.net(x)
