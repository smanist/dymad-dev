import copy
import torch.nn as nn
from typing import Dict, List, Tuple, Union

from dymad.modules.gnn import GNN, ResBlockGNN, IdenCatGNN
from dymad.modules.kernel import KernelScDM, KernelScExp, KernelScRBF, KernelOpSeparable, KernelOpTangent
from dymad.modules.krr import KRRMultiOutputShared, KRRMultiOutputIndep, KRROperatorValued, KRRTangent
from dymad.modules.misc import TakeFirst, TakeFirstGraph
from dymad.modules.mlp import MLP, ResBlockMLP, IdenCatMLP
from dymad.modules.sequential import SimpleRNN, VanillaRNN, StepwiseModel

#: Options for preset neural network models.
NN_MAP = {
    "mlp_smp"  : MLP,
    "mlp_res"  : ResBlockMLP,
    "mlp_cat"  : IdenCatMLP,
    "mlp_1st"  : TakeFirst,
    "gnn_smp"  : GNN,
    "gnn_res"  : ResBlockGNN,
    "gnn_cat"  : IdenCatGNN,
    "gnn_1st"  : TakeFirstGraph,
    "seq_std"  : SimpleRNN,
    "seq_rnn"  : VanillaRNN,
}

def make_network(
    nn_type: str,
    input_dim: int, hidden_dim: int, output_dim: int, n_layers: int,
    seq_len: int = None, **kwargs) -> nn.Module:
    """Factory function to create preset neural network models based on :attr:`~dymad.modules.collections.NN_MAP`.

    Args:
        nn_type (str): Type of network to create.
            One of the keys in :attr:`~dymad.modules.collections.NN_MAP`:
            {'mlp_smp', 'mlp_res', 'mlp_cat', 'mlp_1st',
            'gnn_smp', 'gnn_res', 'gnn_cat', 'gnn_1st', 'seq_std', 'seq_rnn'},
            or 'seq' prefixed versions of MLP and GNN types for sequence models.
        input_dim (int): Dimension of the input features.
        hidden_dim (int): Width of the hidden layers.
        output_dim (int): Dimension of the output space.
        n_layers (int): Number of layers in the network.
        seq_len (int, optional): Length of the input sequences (for sequence-based networks).
        **kwargs: Additional keyword arguments passed to the specific constructors.

    Returns:
        nn.Module: The constructed neural network module.
    """
    _type = nn_type.lower()
    if _type not in NN_MAP:
        if _type[4:] not in NN_MAP:
            raise ValueError(f"Unknown network type '{nn_type}'. Must be one of {list(NN_MAP.keys())}.")
    net_class = NN_MAP.get(_type, None)

    # Special handling for TakeFirst and TakeFirstGraph
    if _type in ["mlp_1st", "gnn_1st"]:
        return net_class(output_dim)

    # Prepare common network options
    net_opts = {
        "input_dim" : input_dim,
        "hidden_dim": hidden_dim,
        "output_dim": output_dim,
        "n_layers"  : n_layers,
    }
    net_opts.update(kwargs)

    # Special handling for sequence models
    if _type[:3] == "seq":
        if _type in NN_MAP:
            # This case always assumes last_only=True
            net_opts["last_only"] = True
            return net_class(seq_len, **net_opts)

        # Extract the base network type (everything after "seq_")
        # This case always assumes last_only=False
        base_type = _type[4:]  # Remove "seq_" prefix
        base_net = make_network(
            nn_type=base_type,
            input_dim=input_dim // seq_len,
            hidden_dim=hidden_dim,
            output_dim=output_dim // seq_len,
            n_layers=n_layers,
            **kwargs
            )
        return StepwiseModel(
            seq_len, net=base_net, last_only=False, **kwargs)

    # Standard handling for MLP and GNN variants
    return net_class(**net_opts)


#: Options for preset autoencoder types.
AE_MAP = {
    "mlp_smp"     : ("mlp_smp", "mlp_smp"),
    "mlp_res"     : ("mlp_res", "mlp_res"),
    "mlp_cat"     : ("mlp_cat", "mlp_1st"),
    "mlp_seq_rnn" : ("seq_rnn", "seq_mlp_smp"),
    "mlp_seq_std" : ("seq_std", "seq_mlp_smp"),
    "mlp_seq_smp" : ("seq_mlp_smp", "seq_mlp_smp"),
    "gnn_smp"     : ("gnn_smp", "gnn_smp"),
    "gnn_res"     : ("gnn_res", "gnn_res"),
    "gnn_cat"     : ("gnn_cat", "gnn_1st"),
    "gnn_seq_rnn" : ("seq_rnn", "seq_gnn_smp"),
    "gnn_seq_std" : ("seq_std", "seq_gnn_smp"),
    "gnn_seq_smp" : ("seq_gnn_smp", "seq_gnn_smp"),
}

def make_autoencoder(
        ae_type: str | Tuple[str, str],
        input_dim: int, hidden_dim: int, latent_dim: int, enc_depth: int, dec_depth: int,
        output_dim: int = None, seq_len: int = None, **kwargs) -> Tuple[nn.Module, nn.Module]:
    """
    Factory function to create preset autoencoder models. Including:

    - [mlp_smp] Simple version: MLP-in MLP-out
    - [mlp_res] Simple version but with ResBlockMLP
    - [mlp_cat] Concatenation as encoder [x MLP(x)], then TakeFirst as decoder
    - [mlp_seq_rnn] RNN-in MLP-out, using a 1-layer unidirectional RNN
    - [mlp_seq_std] RNN-in MLP-out, using standard RNN from pytorch
    - [mlp_seq_smp] MLP-in MLP-out, applied stepwise to sequences
    - The graph version of the above, e.g., gnn_smp, gnn_seq_rnn, etc.

    Args:
        ae_type (str): Type of autoencoder to create.
        input_dim (int): Dimension of the input features.
        hidden_dim (int): Width of the hidden layers.
        latent_dim (int): Dimension of the latent/encoded space.
        enc_depth (int): Number of layers in the encoder.
        dec_depth (int): Number of layers in the decoder.
        output_dim (int, optional): Dimension of the output features, defaults to `input_dim`.
        seq_len (int, optional): Length of the input sequences (for sequence-based autoencoders).
        **kwargs: Additional keyword arguments passed to the specific constructors.
    """
    # Determine encoder and decoder types
    if isinstance(ae_type, str):
        if ae_type.lower() not in AE_MAP:
            raise ValueError(f"Unknown autoencoder type '{ae_type}'. Must be one of {list(AE_MAP.keys())}.")
        enc_type, dec_type = AE_MAP[ae_type.lower()]
    elif isinstance(ae_type, tuple) and len(ae_type) == 2:
        enc_type, dec_type = ae_type
    else:
        raise ValueError(f"Autoencoder type must be a string or a tuple of two strings, got {ae_type}.")

    # Prepare the arguments
    if output_dim is None:
        output_dim = input_dim

    encoder_args = dict(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=latent_dim,
        n_layers=enc_depth,
        seq_len=seq_len
    )
    encoder_args.update(kwargs)
    decoder_args = dict(
        input_dim=latent_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        n_layers=dec_depth,
        seq_len=seq_len
    )
    decoder_args.update(kwargs)

    # Generate the encoder and decoder based on the type
    encoder = make_network(nn_type=enc_type, **encoder_args)
    decoder = make_network(nn_type=dec_type, **decoder_args)

    return encoder, decoder


def _make_scalar_kernel(sk_type: str, input_dim: int, dtype=None, **kwargs) -> nn.Module:
    if sk_type == "rbf":
        return KernelScRBF(in_dim=input_dim, dtype=dtype, **kwargs)
    elif sk_type == "dm":
        return KernelScDM(in_dim=input_dim, dtype=dtype, **kwargs)
    elif sk_type == "exp":
        return KernelScExp(in_dim=input_dim, dtype=dtype, **kwargs)
    else:
        raise ValueError(f"Unknown kernel type '{sk_type}'.")

def make_kernel(k_type: str, input_dim: int, output_dim: int = None, kopts: List=None, dtype=None, **kwargs) -> nn.Module:
    """
    Factory function to create preset kernels. Including:

    - [sc_rbf] Scalar: Radial basis function kernel
    - [sc_dm]  Scalar: Diffusion Map kernel
    - [op_sep] Operator-valued: Separable kernel with multiple scalar kernels

    Args:
        k_type (str): Type of kernel to create.
            One of {'sc_rbf', 'sc_dm'}.
        input_dim (int): Dimension of the input features.
        output_dim (int, optional): Dimension of the output features.
        kopts (List, optional): List of scalar kernel options (for operator-valued kernels).
        dtype: Data type of the kernel parameters.
        **kwargs: Additional keyword arguments passed to the kernel constructors.
    """
    _type = k_type.lower().split('_')
    if len(_type) < 2:
        raise ValueError(f"Unknown kernel type '{k_type}'.")

    if _type[0] == "sc":
        # Scalar-valued kernels
        if output_dim is not None and output_dim != 1:
            raise ValueError(f"Scalar-valued kernel cannot have output_dim>1, got {output_dim}.")
        return _make_scalar_kernel(_type[1], input_dim, dtype=dtype, **kwargs)

    elif _type[0] == "op":
        # Operator-valued kernels
        if output_dim is None or output_dim < 2:
            raise ValueError(f"Operator-valued kernel must have output_dim>=2, got {output_dim}.")

        if _type[1] == "sep":
            if kopts is None or len(kopts) == 0:
                raise ValueError(f"Operator-valued kernel of type 'sep' must have at least one scalar kernel option, got {kopts}.")

            kernels = []
            for _k in kopts:
                _opt = copy.deepcopy(_k)
                _k_type = _opt.pop("type").split('_')[-1]
                _k_input_dim = _opt.pop("input_dim")
                kernels.append(_make_scalar_kernel(_k_type, _k_input_dim, dtype=dtype, **_opt))
            return KernelOpSeparable(kernels, out_dim=output_dim, dtype=dtype, **kwargs)

        if _type[1] == "tan":
            if not isinstance(kopts, dict):
                raise ValueError(f"Operator-valued kernel of type 'tan' must have one and only one scalar kernel, got {kopts}.")

            _opt = copy.deepcopy(kopts)
            _k_type = _opt.pop("type").split('_')[-1]
            _k_input_dim = _opt.pop("input_dim")
            kernel = _make_scalar_kernel(_k_type, _k_input_dim, dtype=dtype, **_opt)
            return KernelOpTangent(kernel, out_dim=output_dim, dtype=dtype, **kwargs)

        else:
            raise ValueError(f"Unknown operator-valued kernel type '{_type[1]}'.")
    else:
        raise ValueError(f"Unknown kernel type '{type}'.")

def make_krr(
        type: str, kernel: Union[Dict, List[Dict]],
        ridge_init=0, jitter=1e-10, dtype=None, device=None) -> nn.Module:
    """
    Factory function to create preset Kernel Ridge Regression (KRR) models. Including:

    - [share] Multi-output KRR with shared scalar kernel
    - [indep] Multi-output KRR with independent scalar kernels
    - [opval] Multi-output KRR with operator-valued kernel
    - [tangent] KRR for vector fields on manifolds

    Args:
        type (str): Type of KRR model to create.
            One of {'krr_shared', 'krr_indep', 'krr_opval', 'krr_tangent'}.
        kernel (Union[Dict, List[Dict]]): Kernel configuration(s).
        ridge_init (float, optional): Initial value for the ridge regularization parameter.
        jitter (float, optional): Jitter added to the diagonal for numerical stability.
    """
    if isinstance(kernel, dict):
        _ker = [kernel]
    elif isinstance(kernel, list):
        _ker = kernel
    else:
        raise ValueError(f"Kernel must be a dictionary or a list of dictionaries, got {type(kernel)}.")

    k_module = []
    for _k in _ker:
        k_type   = _k["type"]
        k_input_dim  = _k["input_dim"]
        k_output_dim = _k.get("output_dim", None)
        k_kopts  = _k.get("kopts", None)
        k_kwargs = {k: v for k, v in _k.items() if k not in {"type", "input_dim", "output_dim", "kopts"}}
        k_module.append(make_kernel(k_type, input_dim=k_input_dim, output_dim=k_output_dim, kopts=k_kopts, dtype=dtype, **k_kwargs))
    if isinstance(kernel, dict):
        # Consistency with input type
        k_module = k_module[0]

    _type = type.lower()
    if _type == "share":
        return KRRMultiOutputShared(kernel=k_module, ridge_init=ridge_init, jitter=jitter, device=device)
    elif _type == "indep":
        return KRRMultiOutputIndep(kernel=k_module, ridge_init=ridge_init, jitter=jitter, device=device)
    elif _type == "opval":
        return KRROperatorValued(kernel=k_module, ridge_init=ridge_init, jitter=jitter, device=device)
    elif _type == "tangent":
        return KRRTangent(kernel=k_module, ridge_init=ridge_init, jitter=jitter, device=device)
    else:
        raise ValueError(f"Unknown KRR type '{type}'.")
